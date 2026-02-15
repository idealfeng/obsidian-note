# 死农专衣物交易平台学习总结（Vue 3 + Express + MySQL）


> 目标：用“前端页面 → API → 数据库 → 部署”的链路，复盘本项目的工程结构、关键实现与踩坑点，方便答辩/复现/后续迭代。  

## 1 目录

- [1. 概述](#1-概述)

- [2. 技术栈](#2-技术栈)

- [3. 总体架构与数据流](#3-总体架构与数据流)

- [4. 代码结构与模块分工](#4-代码结构与模块分工)

- [5. 数据库设计要点](#5-数据库设计要点)

- [6. 后端设计（鉴权与接口）](#6-后端设计鉴权与接口)

- [7. 前端设计（路由、状态、请求）](#7-前端设计路由状态请求)

- [8. 容器化与部署（Docker / Compose / Nginx）](#8-容器化与部署docker--compose--nginx)

- [9. 关键痛点与改进建议](#9-关键痛点与改进建议)

- [10. 导师问答速记（架构 / 痛点 / 安全）](#10-导师问答速记架构--痛点--安全)

## 2 概述

- 业务范围（学习版）：登录/注册、商品列表/分类/详情、收藏、购物车、租赁订单、评论等。

- 三层结构：前端（Vue）负责交互与状态；后端（Express）负责业务与数据访问；数据库（MySQL）负责持久化。

- 部署方式：开发期可“仅 DB 容器化 + 前后端热更新”，验收/生产期使用 `docker-compose.yml` 拉起 `db / backend / frontend`，前端由 Nginx 承载并反代后端。

## 3 技术栈

| 层 | 技术/组件 | 作用 |
| --- | --- | --- |
| 前端 | Vue 3、Vite | 组件化开发与构建/热更新 |
| 状态/路由 | Pinia、Vue Router | 登录态/用户信息/购物车等全局状态；路由与守卫 |
| UI | Element Plus、Sass | 组件库与样式组织 |
| 请求 | Axios（`frontend/src/utils/http.js`） | 统一 baseURL、token 注入、错误拦截 |
| 后端 | Node.js（ESM）、Express | REST API、路由与中间件 |
| 数据访问 | mysql2/promise | 连接池 + 参数化 SQL |
| 数据库 | MySQL 8（utf8mb4） | 表结构、约束、种子数据 |
| 部署 | Docker、Docker Compose、Nginx | 容器化、服务编排、静态站点 + 反向代理 |

## 4 总体架构与数据流

### 4.1 请求链路（开发/生产一致的关键）

核心约定：**前端统一请求 `/api/...`，代理层负责转发与（必要时）去掉 `/api` 前缀，后端路由尽量保持不带 `/api`。**


```text

浏览器

  │  访问静态资源 / 发起 /api 请求

  ▼

Nginx（生产）或 Vite devServer（开发）

  │  /api/* 反代到后端（可 rewrite 去掉 /api）

  ▼

Express 后端（mysql2 访问 MySQL）

  ▼

MySQL

```

### 4.2 登录态与鉴权链路

- 登录/注册：后端生成 `token` 并写入 `users.token`；前端持久化到 localStorage（本项目 key 为 `token_sicau`）。

- 发起请求：Axios 请求拦截器从 localStorage 读取 token，统一加 `Authorization: Bearer <token>`。

- 后端鉴权：从 `Authorization` 解析 token，查库得到 userId（`getUserIdFromToken`），再执行业务 SQL。

## 5 代码结构与模块分工

- `frontend/`

  - `src/router/`：路由表 + 全局守卫（`meta.requiresAuth`）

  - `src/stores/`：Pinia（token、用户信息、购物车等）

  - `src/utils/http.js`：Axios 实例（baseURL、token 注入、错误拦截）

  - `src/apis/`：按业务域拆分 API 调用封装（收藏/评论/订单等）

- `backend/`

  - `server.js`：Express 入口、路由与 MySQL 连接池（学习阶段集中在一个文件）

- `db/init.sql`

  - 建库、建用户、建表、初始化分类/商品等种子数据

我学到的拆分原则：前端保持 **API（请求）/ Store（状态）/ View（页面）** 三层；后端上线更建议拆成 `routes/ + services/ + db/`（避免 `server.js` 过长、难测、难复用）。

## 6 数据库设计要点

主要表（统一 `utf8mb4`）：

- `users`：用户名、密码哈希、token、头像

- `categories`、`goods`：分类与商品

- `favorites`：收藏（`(user_id, product_id)` 唯一约束）

- `cart`：购物车（`(user_id, product_id)` 唯一约束；含 `quantity/selected`）

- `rental_orders`：租赁订单（含 `status` 枚举、日期区间、金额字段）

- `comments`：评论（可选星级）

我学到的点：

- 字符集统一用 `utf8mb4`，避免中文/emoji 存储异常。

- “同一用户同一商品只允许一条记录”的需求优先用 **唯一索引** 落库，减少并发下的逻辑漏洞（收藏、购物车）。

- 当前 `goods.id` 是 `INT`，而部分业务表里的 `product_id` 是 `VARCHAR`：学习阶段先记录风险，后续应统一类型，降低 JOIN/对齐成本。

## 7 后端设计（鉴权与接口）

### 7.1 鉴权方式

- token 生成：登录/注册成功后生成并写入 `users.token`。

- token 传递：客户端请求头 `Authorization: Bearer <token>`。

- token 校验：后端解析 token 后查库得到 userId，再执行后续业务逻辑。

安全实现要点：主要 SQL 使用 `pool.execute(sql, params)` 参数化，避免 SQL 注入（代码中也保留了“危险拼接”的对照示例）。

### 7.2 接口清单（按业务域）

（以 `backend/server.js` 为准）

| 业务域 | 方法 | 路径 | 说明 |
| --- | --- | --- | --- |
| 登录 | POST | `/login` | 登录/注册合并入口，返回 token + userInfo |
| 用户 | GET | `/profile` | 获取当前用户资料 |
| 收藏 | GET | `/favorites` | 获取收藏列表 |
| 收藏 | POST | `/favorites` | 添加收藏 |
| 收藏 | DELETE | `/favorites/:id` | 取消收藏 |
| 购物车 | GET | `/cart` | 获取购物车 |
| 购物车 | POST | `/cart` | 添加商品或累计数量 |
| 购物车 | PUT | `/cart/select-all` | 全选/全不选 |
| 购物车 | PUT | `/cart/:productId` | 修改数量/选中状态 |
| 购物车 | DELETE | `/cart/batch` | 批量删除 |
| 购物车 | DELETE | `/cart/:productId` | 删除单项 |
| 租赁 | GET | `/rentals` | 获取租赁订单 |
| 租赁 | POST | `/rentals` | 创建租赁订单 |
| 租赁 | PUT | `/rentals/:orderId/cancel` | 取消订单 |
| 租赁 | DELETE | `/rentals/:orderId` | 删除订单 |
| 评论 | GET | `/products/:productId/comments` | 获取商品评论 |
| 评论 | POST | `/products/:productId/comments` | 发表评论 |
| 评论 | DELETE | `/comments/:commentId` | 删除评论 |

## 8 前端设计（路由、状态、请求）

### 8.1 路由守卫（登录态控制）

- `frontend/src/router/index.js`：对 `meta.requiresAuth` 的路由做拦截，没有 token 则跳回登录页，并带上 `redirect`。

### 8.2 状态管理（Pinia）

- token 与用户信息落在 store，并持久化到 localStorage。

- key 统一（如 `token_sicau`）非常重要：守卫、拦截器、store 三者必须一致，否则会出现“登录后刷新丢状态”的问题。

### 8.3 Axios 统一封装（请求拦截与错误处理）

- `frontend/src/utils/http.js`：优先从 localStorage 读 `token_sicau`，统一注入 `Authorization`。

- 响应拦截器预留 401 统一处理入口（清 token、跳转登录、提示等）。

## 9 容器化与部署（Docker / Compose / Nginx）

### 9.1 Docker 的作用（为什么要容器化）

- 环境一致性：Node/MySQL/Nginx 版本与运行方式固定到镜像。

- 隔离解耦：前端/后端/数据库依赖互不污染，故障可单独定位与重启。

- 交付友好：把“怎么跑起来”固化成 `docker-compose.yml`，便于验收复现。

### 9.2 两种启动方式（开发 vs 验收/部署）

1）本地热更新（学习/开发更顺手）：只把 DB 放进容器，前后端在宿主机运行。

```bash

docker compose up -d db

  

cd backend

npm i

npm run dev

  

cd ../frontend

npm i

npm run dev -- --port 5175

```

2）全容器启动（更接近验收/部署）：`db + backend + frontend` 全部交给 Compose。

```bash

docker compose up -d --build

docker compose ps

```

默认验收入口（参考 `.env.sample`）：

- 前端：`http://localhost:5173`

- 后端：`http://localhost:15200`（容器内 `5200` 映射到宿主机 `15200`）

- MySQL：`127.0.0.1:13306`

### 9.3 Compose 里的关键机制

- 数据库初始化：`db/init.sql` 挂载到 `/docker-entrypoint-initdb.d/`，首次启动自动建库建表与种子数据。

- 数据持久化：`mysql-data` volume 保存 `/var/lib/mysql`，容器重建数据仍在。

- 启动顺序：db healthcheck + backend depends_on（db healthy 后再启动后端）。

- 容器网络：容器内服务通过 service name 访问（如 `DB_HOST=db`）。

### 9.4 Nginx 的作用（生产发布与同源访问）

- 静态资源服务：承载 `dist/`。

- SPA 回退：`try_files ... /index.html`，避免前端路由刷新 404。

- API 反代：把 `/api/` 转发到 `backend:5200`，实现前后端同源部署，降低跨域与部署复杂度。

## 10 关键痛点与改进建议

- 数据库未就绪导致后端启动失败：Compose healthcheck + depends_on；代码侧可补连接重试与降级提示。

- `/api` 前缀一致性：统一“前端永远 `/api/...`，代理层去前缀，后端不带 `/api`”；避免出现路由错位（例如后端写了 `/api/ping` 但代理已去掉 `/api`）。

- 健康检查脚本对不上：`check.sh` 访问 `/api/health`，但后端未实现；应改为真实存在的接口或补一个健康检查路由。

- Compose YAML 缩进风险：`backend.volumes` 缩进不规范会导致解析失败，应按列表缩进格式整理。

- ID 类型不统一（`INT` vs `VARCHAR`）：统一主键/外键类型；补齐索引与外键约束。

- 代码组织：后端建议拆分路由与服务层；前端 `src/apis/` 与 `src/stores/` 内容保持“职责单一”，避免文件混入不同层代码。

## 11 导师问答速记（架构 / 痛点 / 安全）

### 11.1 架构设计

  

- Q：为什么拆成前端/后端/数据库三层？

  - A：职责分离，便于维护、扩展、独立部署；前端聚焦交互，后端聚焦业务与数据访问，数据库聚焦持久化与约束。

- Q：为什么生产要用 Nginx？

  - A：静态资源分发更高效；支持 SPA 回退；可把 `/api` 反代到后端，做到同源部署、跨域成本更低。


### 11.2 关键痛点与应对（答辩用短句）

  
- “后端连不上库”：healthcheck + depends_on +（可选）连接重试。

- “开发/生产接口路径不一致”：统一 `/api` 约定，代理层处理前缀。

- “登录态不稳定”：token key 常量化；拦截器优先读 localStorage。

### 11.3 安全防范（当前实现 + 可提升点）

- 密码：使用 `bcrypt`/`bcryptjs` 哈希存储与比对，避免明文落库；避免日志打印敏感信息。

- SQL 注入：使用 `pool.execute(sql, params)` 参数化；建议补输入校验与字段长度限制。

- Token 风险：token 存 localStorage 容易受 XSS 影响；建议升级 JWT（过期/刷新）或 HttpOnly Cookie + SameSite，并配合 CSP 与输出转义。

- CORS：当前 `origin: '*'` 便于学习但不适合生产；应限制允许来源域名。

- 接口防刷：对登录等接口加限流（如 `express-rate-limit`），并加安全响应头（如 `helmet`）。

- 配置与密钥：避免提交真实 `.env`；生产使用最小权限账号与安全的密钥注入方式（CI/Docker secrets）。