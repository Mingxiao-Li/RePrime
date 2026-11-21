# 部署到 GitHub Pages（完整步骤）

## 部署状态（2026-11-21 已完成）

- 线上地址：**https://mingxiao-li.github.io/RePrime/**
- 仓库：`Mingxiao-Li/RePrime`，**public**，默认分支 **`master`**
- Pages：来源 `master` / `/docs`，状态 `built`，HTTPS 已强制
- 已验证：页面、图片、两段视频（含 AAC 音频）、Code 链接、移动端、axe WCAG 2A/AA
- 本文保留为后续更新 / 迁移 / 重新部署的参考

> 历史说明：部署前仓库是 private、Pages 未开启。下面 A0–A4 记录了从零开始的完整流程，供以后换仓库或迁移时复用。

---

# 方案 A：RePrime 仓库 + `docs/`（推荐）

## A0. 把 RePrime 改成 public

1. 打开 `https://github.com/Mingxiao-Li/RePrime`
2. **Settings → General**
3. 拉到最下面 **Danger Zone**
4. **Change visibility → Change to public**
5. 按提示输入仓库名确认

> 当前仓库里只有 README / LICENSE，风险很低。如果以后要放代码，注意不要提交 token、密钥、私有数据或过大的 checkpoint。

## A1. 克隆仓库

```bash
cd ~/Documents
git clone https://github.com/Mingxiao-Li/RePrime.git
cd RePrime
```

已经克隆过：

```bash
cd ~/path/to/RePrime
git pull
```

## A2. 复制页面到 `docs/`

注意结尾的 `/.`，它会把 `.nojekyll` 一起复制：

```bash
mkdir -p docs
cp -R /path/to/reprime_project_page/. docs/
```

检查：

```bash
ls -la docs
```

应该看到：

```
index.html
.nojekyll
assets/
README.md
DEPLOY.md
```

如果仓库里本来就有 `docs/`，先看 `git status`，不要覆盖已有内容。

## A3. 提交并推送（注意分支是 `master`）

```bash
git add docs
git commit -m "Add RePrime project page"
git push origin master
```

## A4. 开启 Pages

1. 仓库 → **Settings → Pages**
2. **Build and deployment → Source**：`Deploy from a branch`
3. **Branch**：`master`，文件夹：**`/docs`**
4. **Save**
5. 等 1–3 分钟，上方出现：

```
Your site is live at https://mingxiao-li.github.io/RePrime/
```

6. 打开验证。

## A5. 以后更新

```bash
git add docs
git commit -m "Update project page"
git push origin master
```

Pages 自动重建，1–2 分钟生效；浏览器缓存用 `Cmd + Shift + R` 强刷。

---

# 方案 B：独立 public 仓库 `reprime-page`

适合代码仓库暂时不公开的情况。

## B1. 新建仓库

- 仓库名：`reprime-page`
- 可见性：**Public**
- 不要勾 “Add a README file”

## B2. 推送页面

```bash
cd /path/to/reprime_project_page

git init
git add .
git commit -m "RePrime project page"
git branch -M main
git remote add origin https://github.com/Mingxiao-Li/reprime-page.git
git push -u origin main
```

## B3. 开启 Pages

Settings → Pages → Source: `Deploy from a branch` → Branch: `main` / **`/ (root)`** → Save。

地址：

```
https://mingxiao-li.github.io/reprime-page/
```

## B4. 处理 Code 按钮

`RePrime` 还是 private 时，页面上的 Code 会 404。两个选择：

1. 先把 Code 按钮改成占位（我可以帮你改）：
   ```html
   <span class="btn btn-outline is-disabled" aria-disabled="true">Code · coming soon</span>
   ```
2. 等代码仓库公开后，再换回 `https://github.com/Mingxiao-Li/RePrime`。

---

# 认证（你已经登录，通常不用做）

检查：

```bash
gh auth status
```

应该显示 `Logged in to github.com account Mingxiao-Li`。

如果 `git push` 仍提示输入密码：

```bash
gh auth setup-git
```

然后再 push。也可以使用 PAT（`repo` scope）作为密码，或改用 SSH。

---

# 部署后验证清单

- [ ] `https://mingxiao-li.github.io/RePrime/` 能打开，不是 404
- [ ] 导航栏各 section 能跳转
- [ ] DevTools → Network 没有 404
- [ ] 两段视频能播放，点击后**有声音**，controls 能静音
- [ ] 手机宽度（390px）无横向滚动
- [ ] 无痕窗口打开正常
- [ ] Settings → Pages → **Enforce HTTPS** 已勾选
- [ ] 页面里搜不到 `CVPR` / `under review` / `submission`
- [ ] Code 按钮指向的仓库是 public（否则会 404）

## 更新社交预览图

把 `index.html` 里的相对路径改成绝对 URL：

```html
<meta property="og:image" content="https://mingxiao-li.github.io/RePrime/assets/img/og.jpg">
```

方案 B 的话换成：

```html
<meta property="og:image" content="https://mingxiao-li.github.io/reprime-page/assets/img/og.jpg">
```

---

# 常见问题

### 访问 404

1. Pages 的 Branch / Folder 选错：方案 A 是 `master` + `/docs`；方案 B 是 `main` + `/ (root)`。
2. `docs/index.html` 或根目录 `index.html` 不存在。
3. 仓库是 private 且是 Free 计划 → 改 public，或走方案 B。
4. 刚 Save，构建还没完成 → 等 1–3 分钟，或看 **Actions**。
5. 仓库里已有自定义 Pages workflow → Source 改用 **GitHub Actions**。

### Code 按钮 404

`Mingxiao-Li/RePrime` 还是 private。把仓库改成 public，或先把按钮换成 “Code · coming soon”。

### CSS / JS / 图片 404

- 路径区分大小写，`assets/img/teaser.jpg` 必须完全一致。
- 确认整个 `assets/` 都上传了。
- 确认 `.nojekyll` 存在。
- `git ls-files docs | head` 检查文件是否进了仓库。

### 视频不播放

- 当前是 H.264 + AAC MP4，主流浏览器支持。
- Network 里看 `assets/video/reprime_method.mp4` 是否 200；404 就是没上传。
- 校园网/公司网可能拦截媒体流。

### 改了但线上没变

- 等 1–2 分钟。
- `Cmd + Shift + R` 强刷。
- 看 Actions 最近一次 `pages build and deployment` 是否成功。

### 推送失败 / 仓库太大

- 当前整包约 5.7 MB，远低于限制（单文件 100 MB，仓库建议 < 1 GB）。
- 大视频建议压到 1080p 以下；一般不需要 LFS。

---

# 可选：自定义域名

1. DNS：
   - `www` → CNAME → `mingxiao-li.github.io`
   - 根域名 → A → `185.199.108.153`、`185.199.109.153`、`185.199.110.153`、`185.199.111.153`
2. Settings → Pages → **Custom domain** 填域名 → Save。
3. DNS 生效后勾选 **Enforce HTTPS**。
4. 仓库生成的 `CNAME` 文件不要删。

---

# 部署完成后的建议

- 代码仓库 **About → Website** 填线上地址。
- 在代码仓库 README 顶部加一行：

```markdown
**Project page:** https://mingxiao-li.github.io/RePrime/
```

- 打 tag：

```bash
git tag -a page-v1 -m "Project page v1"
git push origin page-v1
```

- 如果还在双盲评审期：页面公开 + 代码仓库 public 会暴露身份。请确认这是你想要的；否则先走方案 B，代码仓库保持 private。
