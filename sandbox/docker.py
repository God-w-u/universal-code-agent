"""DockerSandbox：在一次性容器里执行命令。

和 LocalSandbox 是同一个接口的两种后端：调用方拿着 Sandbox 用就行，
不关心命令是在宿主机上跑还是在容器里跑。只用 Docker CLI（不引 Docker
SDK），每次 run 起一个临时容器，跑完就没了。

设计取舍：

- **临时容器**：每次执行一个 `docker run --rm`，容器名带随机后缀，
  绝不复用。正常结束由 `--rm` 在 daemon 那边删掉；被我们超时打断的
  那条路 `--rm` 管不到（容器还活着），由 `docker rm -f` 点名杀掉——
  两条路都收干净。
- **映射**：host workspace 本身以 bind mount 挂在容器里的 /workspace
  （挂进去的永远只有 workspace 这一个目录），容器里的 cwd 由 host cwd
  相对 workspace 的位置推出来。cwd 必须先过 base.resolve_cwd（落在
  workspace 以内），所以映射不出 workspace——和 LocalSandbox 是同一份
  边界检查。
- **镜像**：构造时显式给 image，本地没有就报错——不自动 pull，
  `--pull=never` 也堵住 docker run 自己那点“顺手拉一下”的念头。
- **env**：只传调用方显式给的键值（`-e`），不把宿主机环境变量倒进
  容器（docker 的默认行为本来就不继承宿主机环境，这里也不去“补齐”）。
- **不静默退回 local**：docker 不在、daemon 没起、镜像没有，都抛
  ToolError 说清楚，绝不换成本机执行。
"""

import base64
import shutil
import uuid
from pathlib import Path

from sandbox.base import CommandResult, Sandbox, check_run_args, resolve_cwd
from tools.base import ToolError
from tools.paths import check_root
from tools.run_command import DEFAULT_TIMEOUT, decode_output, run_process

# workspace 在容器里的固定挂载点。命令里写 plain "docker"，让 shell
# 自己去找；shutil.which 只负责回答“这台机器有没有 docker”。
DOCKER = "docker"
CONTAINER_ROOT = "/workspace"

# 容器名前缀：清理和排查时一眼认得出是自己起的
NAME_PREFIX = "uac-sandbox-"

# 探测 docker（version / image inspect）和兜底清理最多各等多久
PROBE_TIMEOUT = 15.0
CLEANUP_TIMEOUT = 15.0


def first_line(raw: bytes) -> str:
    """把 Docker 的报错压成一行：它的话经常一屏，捡开头那句最像原因。"""
    text = decode_output(raw, "Docker 的错误输出").strip()
    if not text:
        return "（Docker 没有给出原因）"
    return text.splitlines()[0].strip()


def payload(command: str) -> str:
    """把命令包成一段「谁看都是普通字符」的容器内 sh 脚本。

    为什么要绕 base64 这一圈：这条 docker 命令最后要过宿主机 shell
    （Windows 是 cmd、别处是 sh），而命令本身常带引号（比如
    `"C:/.../python.exe" "x.py"`）——直接塞进 `sh -c "…"` 会两三层
    引号互相咬。base64 只有字母数字和 +/=，没有空格、引号、管道，
    穿哪一层都原样到达容器，进容器再解开交给 sh 执行。
    """
    encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
    return f"echo {encoded} | base64 -d | sh"


class DockerSandbox(Sandbox):
    """在一次性容器里执行命令。

    image 必须显式给（keyword-only，没有默认值）：用哪个镜像跑是调用方
    的决定，不给一个猜的默认值。workspace / timeout 的含义和
    LocalSandbox 完全一致。
    """

    kind = "docker"

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        image: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout 必须大于 0，收到 {timeout}")

        if not isinstance(image, str) or not image.strip():
            raise ToolError(
                "docker sandbox 必须指定 image（不会自动拉取），例如 "
                "DockerSandbox(image='python:3.12-slim')"
            )

        self._workspace = check_root(workspace)
        self.image = image.strip()
        self.timeout = timeout

        if shutil.which(DOCKER) is None:
            raise ToolError(
                "没有找到 docker 命令：装好 Docker 并保证 docker 在 PATH 上，"
                "或者把 sandbox 换回 local（这条路不会自己退回本机执行）"
            )

        # daemon 和镜像等第一次真要跑的时候再验，验过一次就记住：
        # 构造可能发生在只配置不执行的场合，而 daemon 状态是会变的。
        self._checked = False

    @property
    def workspace(self) -> Path:
        return self._workspace

    def run(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """在容器里执行一条命令。

        返回的 CommandResult.command 是**调用方给的原命令**（不是拼出来的
        那条 docker 命令行）：结果文本是给模型看的，它认识的应该是自己写
        的那条命令。cwd 记的是 host 路径，和 LocalSandbox 的含义一致。
        """
        seconds = check_run_args(command, timeout, self.timeout)
        target = resolve_cwd(cwd, self._workspace)
        self._check_ready()

        name = NAME_PREFIX + uuid.uuid4().hex[:12]
        docker_command = self._docker_command(command, target, env, name)

        # 起进程还是那份 run_process：超时准时杀掉、两路输出都收——
        # 那几个 Windows 坑的注释在 tools/run_command.py 里它头上。
        code, out_raw, err_raw = run_process(
            docker_command, cwd=self._workspace, timeout=seconds
        )
        if code is None:
            # 被我们打断的路：容器还在跑，--rm 要等它自己结束才生效，
            # 这里点名杀掉再删，别让它留到后台去
            self._remove_container(name)
            return CommandResult(
                command=command,
                cwd=target,
                exit_code=None,
                stdout="",
                stderr="",
                timeout=seconds,
                timed_out=True,
            )

        # docker run 的退出码就是容器里命令的退出码；docker 自己出错
        # （比如执行到一半 daemon 挂了）是 125/126/127，也照常返回，
        # 报错就在标准错误里——「非 0 是结果不是异常」的规矩不变。
        return CommandResult(
            command=command,
            cwd=target,
            exit_code=code,
            stdout=decode_output(out_raw, "标准输出"),
            stderr=decode_output(err_raw, "标准错误"),
            timeout=seconds,
            timed_out=False,
        )

    def _check_ready(self) -> None:
        """第一次执行前确认 daemon 在、镜像在本地；验过一次就记住。

        失败**不缓存**：daemon 可能刚被启动、镜像可能刚被拉到，
        下次 run 要重新试，不能把一次失败记成「永远不行」。
        """
        if self._checked:
            return

        code, _, err = run_process(
            f"{DOCKER} version", cwd=self._workspace, timeout=PROBE_TIMEOUT
        )
        if code != 0:
            raise ToolError(
                f"Docker daemon 不可用（docker version 失败）：{first_line(err)}。"
                "这条 sandbox 不会退回 local，先把 Docker 起起来再试"
            )

        code, _, err = run_process(
            f'{DOCKER} image inspect "{self.image}"',
            cwd=self._workspace,
            timeout=PROBE_TIMEOUT,
        )
        if code != 0:
            raise ToolError(
                f"本机没有镜像 {self.image!r}，而这个 sandbox 不会自动 pull："
                f"{first_line(err)}。先 docker pull {self.image} 再试"
            )

        self._checked = True

    def _docker_command(
        self,
        command: str,
        cwd: Path,
        env: dict[str, str] | None,
        name: str,
    ) -> str:
        """拼出宿主机要跑的那条 docker 命令（测试直接拿它验形状）。"""
        parts = [
            DOCKER,
            "run",
            "--rm",  # 正常结束的容器由 daemon 删
            "--pull=never",  # 本地没有镜像就失败，不许偷偷拉
            f"--name {name}",
            f'-v "{self._workspace.as_posix()}:{CONTAINER_ROOT}"',
            f"-w {self._container_path(cwd)}",
        ]
        # 只放调用方显式给的变量；env 为 None 就一个 -e 都没有，
        # 容器拿到的环境就是镜像自带的那份
        for key, value in sorted((env or {}).items()):
            parts.append(f'-e "{key}={value}"')
        parts += [self.image, "sh", "-c", f'"{payload(command)}"']
        return " ".join(parts)

    def _container_path(self, host_path: Path) -> str:
        """host 路径 → 容器里的路径。

        workspace 本身就是 /workspace，子目录按相对位置接上去。传进来
        的 cwd 已经过 resolve_cwd（必须落在 workspace 以内），所以这里
        推不出 workspace——挂进容器的也只有 workspace 这一个目录。
        """
        relative = host_path.relative_to(self._workspace).as_posix()
        if relative == ".":
            return CONTAINER_ROOT
        return f"{CONTAINER_ROOT}/{relative}"

    def _remove_container(self, name: str) -> None:
        """超时后收拾容器：docker rm -f 先杀掉还在跑的，再删掉。

        失败不抛：命令已经超时这件事比「清理没成功」更该报给调用方，
        别让清理的异常盖住它。清理扑空的极端情况（容器还没来得及建出来
        就被我们打断）也不要紧：`--rm` 在 daemon 那边记着，真建出来了
        跑完也会自己没。
        """
        run_process(
            f"{DOCKER} rm -f {name}", cwd=self._workspace, timeout=CLEANUP_TIMEOUT
        )
