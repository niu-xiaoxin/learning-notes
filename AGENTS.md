# AI 助手操作指南

本仓库是用户的个人学习笔记库。以下是关键操作流程，请严格遵循。

## 仓库结构概览

```
learning-notes/
├── AGENTS.md              # 本文件：AI 操作指南
├── README.md
├── 学习日志.md
├── 学习规划/
├── 数学基础/
├── 编程与工具/
└── 深度学习与AI/
    ├── README.md          # 【需更新】笔记入口索引
    └── nn-zero-to-hero/   # Karpathy 课程主目录
        ├── 课程大纲.md    # 【需更新】课程索引与目录说明
        ├── lectures/      # 官方课程代码（勿修改）
        ├── 参考资料/      # 配套项目源码
        └── 笔记/          # 【笔记存放位置】
```

## 新增学习笔记的完整流程

当用户提供新的课程笔记内容时，必须按以下 3 步完成所有更改：

### 第 1 步：保存笔记文件

将笔记保存到 `深度学习与AI/nn-zero-to-hero/笔记/` 目录下。

命名规范：`{课程名}_{主题}_课程笔记.md`

示例：
- `micrograd_课程笔记.md`
- `makemore_课程笔记.md`
- `makemore_part2_MLP_课程笔记.md`

### 第 2 步：更新两个索引文件

**文件 A：`深度学习与AI/README.md`**

在 `## 课程笔记` 表格中追加一行：

```markdown
| [新笔记标题](nn-zero-to-hero/笔记/新文件名.md) | 一句话简介 |
```

**文件 B：`深度学习与AI/nn-zero-to-hero/课程大纲.md`**

在 `## 目录说明` 的目录树中，`笔记/` 下追加新文件名：

```
└── 笔记/
    ├── ...已有笔记...
    └── 新文件名.md        ← 追加这一行
```

### 第 3 步：提交并推送到 GitHub

```bash
cd D:\learning-notes
git add -A
git commit -m "添加 xxx 课程笔记"
git push origin master
```

## 操作检查清单

每次新增笔记后，确认以下 3 项全部完成：

- [ ] `nn-zero-to-hero/笔记/` 下有新的 .md 文件
- [ ] `深度学习与AI/README.md` 的笔记表格中有新行
- [ ] `深度学习与AI/nn-zero-to-hero/课程大纲.md` 的目录树中有新文件名

缺一不可，否则索引会和实际文件不一致。

## 注意事项

- `lectures/` 目录是从 GitHub 克隆的官方代码，不要修改
- `参考资料/` 存放配套项目源码，不要修改
- 笔记文件内容由用户提供，直接保存即可，不要擅自修改内容
- 提交前用 `git status` 确认变更文件数量符合预期（通常是 3 个文件）
- 本仓库的远程分支是 `master`（不是 main）
- GitHub 用户名：niu-xiaoxin
- 仓库地址：https://github.com/niu-xiaoxin/learning-notes
