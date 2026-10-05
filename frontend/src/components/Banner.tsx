/**
 * 顶上那条提示：出错时冒出来，点「知道了」收掉。
 *
 * 只放**需要用户看一眼**的事（这一轮失败、取消失败、会话没了）。像「工具调用
 * 失败」这种属于某一轮过程里的事，记在那一轮下面就行，不该弹到这儿——横幅一
 * 多，用户就不看了。
 *
 * role="alert"：读屏软件会念出来（这是提示，不是装饰）。
 */

export interface BannerProps {
  text: string;
  onDismiss: () => void;
}

export function Banner({ text, onDismiss }: BannerProps) {
  return (
    <div className="banner" role="alert">
      <span className="banner-icon" aria-hidden="true">
        ⚠
      </span>
      <span className="banner-text">{text}</span>
      <button type="button" onClick={onDismiss}>
        知道了
      </button>
    </div>
  );
}
