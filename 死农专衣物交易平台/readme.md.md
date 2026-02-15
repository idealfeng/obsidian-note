# 死农专衣物交易平台（Vue3 + Express + MySQL）学习总结

  

> 面向：复盘本项目从“前端页面 → API → 数据库 → 部署”的完整链路，沉淀可复用的工程化做法与踩坑经验。

  

## 1. 项目做了什么

  

- 前端：Vue 3 + Vite + Pinia + Vue Router + Element Plus，包含登录、首页、分类、详情、收藏、购物车、订单/租赁等页面。

- 后端：Node.js（ESM）+ Express + mysql2/promise，提供登录/注册、用户资料、收藏、购物车、租赁订单、评论等接口。

- 数据库：MySQL 8（utf8mb4），通过 `db/init.sql` 初始化库、用户、表结构与部分种子数据。

- 部署：`docker-compose.yml` 一键拉起 `db / backend / frontend`；生产前端由 Nginx 承载并反代后端。

  

## 2. 目录与模块分工（我学到的拆分方式）

  

- `frontend/`

  - `src/router/`：路由表与全局守卫（鉴权控制）

  - `src/stores/`：Pinia 状态（token、用户信息、购物车等）

  - `src/utils/http.js`：Axios 实例 + 拦截器（统一挂载 `Authorization: Bearer <token>`）

  - `src/apis/`：按业务域拆 API（如收藏、评论、订单…）

- `backend/`

  - `server.js`：Express 入口、路由与 MySQL 连接池

- `db/init.sql`

  - 建库、建用户、建表、初始化分类与商品数据

  

核心体会：前端 **API / Store / View** 三层拆分能明显降低耦合；后端用 **路由分组 + 数据访问层（DAO/Service）** 会更易维护（本项目后端集中在 `server.js`，适合学习阶段，但上线建议拆分）。

  

## 3. 数据库设计要点（从 init.sql 复盘）

  

主要表（均为 `utf8mb4`）：

  

- `users`：用户名、密码哈希、token、头像

- `categories`、`goods`：分类与商品

- `favorites`：收藏（`(user_id, product_id)` 唯一约束）

- `cart`：购物车（`(user_id, product_id)` 唯一约束，含 `quantity/selected`）

- `rental_orders`：租赁订单（含 `status` 枚举、日期区间、金额字段）

- `comments`：评论（含可选星级）

  

我学到的点：

  

- 字符集统一用 `utf8mb4`，避免 emoji/中文存储问题。

- 需要“一个用户同一商品只能有一条记录”的地方，用 **唯一索引** 比业务代码更可靠（如收藏、购物车）。

- `goods.id` 是 `INT`，而部分业务表里 `product_id` 是 `VARCHAR`；这会导致对齐与 JOIN 更麻烦，后续应统一类型（学习阶段先记录问题）。

  

## 4. 后端接口与鉴权（从 server.js 复盘）

  

### 4.1 鉴权方式

  

- 登录/注册成功后生成 `token`，写入 `users.token`

- 前端请求带 `Authorization: Bearer <token>`

- 后端通过 token 查用户 id（`getUserIdFromToken`）实现鉴权

  

学习点：使用 `pool.execute(sql, params)` 的 **预编译参数**，避免 SQL 注入（代码里也保留了“危险拼接”示例作为对比）。

  

### 4.2 主要接口（按功能归类）

  

（路径来自 `backend/server.js`）

  

- 登录/注册：`POST /login`

- 用户信息：`GET /profile`

- 收藏：`GET /favorites`、`POST /favorites`、`DELETE /favorites/:id`

- 购物车：`GET /cart`、`POST /cart`、`PUT /cart/select-all`、`PUT /cart/:productId`、`DELETE /cart/batch`、`DELETE /cart/:productId`

- 租赁：`GET /rentals`、`POST /rentals`、`PUT /rentals/:orderId/cancel`、`DELETE /rentals/:orderId`

- 评论：`GET /products/:productId/comments`、`POST /products/:productId/comments`、`DELETE /comments/:commentId`

  

## 5. 前端路由与请求链路（我最重要的收获）

  

### 5.1 路由守卫（登录态控制）

  

- `frontend/src/router/index.js`：对 `meta.requiresAuth` 的路由做拦截，没有 token 则跳回登录页，并带上 `redirect`。

  

### 5.2 Axios 拦截器（统一加 token）

  

- `frontend/src/utils/http.js`：每次请求优先从 `localStorage` 读 `token_sicau`，再设置 `Authorization` 头。

- 响应拦截器对 401 预留了统一处理入口（清 token、跳转登录等）。

  

### 5.3 /api 前缀的处理（开发与生产一致）

  

- 开发（Vite）：`frontend/vite.config.js` 用代理把 `/api/*` 转发到 `http://localhost:5200`，并 rewrite 去掉 `/api` 前缀。

- 生产（Nginx）：`frontend/nginx.conf` 将 `/api/` 反代到 `http://backend:5200/`（同样会去掉 `/api` 前缀）。

  

结论：**前端统一请求 `/api/...`，后端路由建议统一为不带 `/api` 前缀**（由代理负责“去前缀”），这样开发/生产的路径行为一致。

  

## 6. Docker / Compose（学到的工程化点）

  

- `docker-compose.yml`：三个服务一起编排，并用 `depends_on + healthcheck` 等待 db 就绪再启动后端。

- `backend/Dockerfile`：两阶段构建（deps → runtime），并保留了依赖复制与调试输出，便于定位容器内依赖缺失问题。

- `frontend/Dockerfile`：构建 `dist` 后用 Nginx 承载静态资源，并通过 `nginx.conf` 反代 `/api/`。

  

## 7. 我记录下来的“踩坑/待改进”

  

这些点不影响学习结论，但适合后续清理完善：

  

- `docker-compose.yml` 中 `backend.volumes` 的缩进需要修正为列表缩进，否则可能导致 compose 解析失败。

- `check.sh` 访问的是 `/api/health`，但后端当前未实现该路由；建议改成真实存在的健康检查接口，或在后端补一个 `/health`。

- `backend/server.js` 中存在 `/api/ping` 路由：由于前端/NGINX 都会把 `/api` 去掉，访问 `/api/ping` 最终会落到后端的 `/ping`（不匹配）。建议统一成 `/ping` 或调整代理策略。

- `frontend/src/apis/user.js` 文件内容混入了 store 代码，建议整理为“API 文件只放请求函数，store 只放状态与 action”。

  

## 8. 下一步可以怎么练（可选）

  

- 后端拆层：把 `server.js` 拆成 `routes/ + services/ + db/`，并封装统一的错误处理与返回格式。

- 鉴权升级：把 token 改成 JWT（含过期时间与刷新机制），并增加权限/角色字段演练 RBAC。

- 数据一致性：统一 `product_id` 类型（建议与 `goods.id` 一致），并逐步补齐外键约束与索引策略。

- 可观测性：给 API 增加请求日志、慢查询日志、统一 request-id 便于排查。