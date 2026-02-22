# ZinnyWorld Website Specification (Pre-Coding)

## 1) Product Vision
ZinnyWorld is a direct-to-consumer e-commerce website for human hair products (wigs, frontals, closures), built on a plain PHP + MySQL stack with API-ready structure for future mobile app integration.

## 2) Customer-Facing Scope (MVP)
- Home page (hero, featured categories, promo, trust signals)
- Shop page with filters and sorting
- Product details with variant selection and media gallery
- Cart + checkout
- Guest checkout with account upsell
- Customer account (orders, addresses, profile)
- Reviews and ratings
- Discount code support

## 3) Admin Scope (MVP)
- Admin authentication and role controls (start with admin/editor)
- Product catalog CRUD with variants
- Variant-level stock management
- Upload multiple product images and videos
- Order management and status updates
- Customer management overview
- Inventory and low-stock report widgets

## 4) Variant Model
- Product types: wigs, frontals, closures
- Lengths (inches): 10, 14, 18, 20, 22, 24, 28, 30, 32+
- Texture: virgin hair
- Colors: mostly black (extensible)
- Density examples:
  - Curly: 300g
  - Short straight: 250g
- Lace: HD lace

## 5) Payments & Shipping
- Payment methods:
  - PayPal
  - Bank transfer
- Shipping regions:
  - Australia (default)
  - Nigeria (occasional bundles)

## 6) Technical Architecture (Plain PHP + MySQL 7+)
Recommended layered structure:
- `public/` for entrypoints and static assets
- `app/Controllers/` request handlers
- `app/Services/` business logic
- `app/Repositories/` DB interaction
- `app/Models/` entities
- `routes/` route maps
- `config/` environment and integrations
- `storage/` uploads/logs
- `api/` versioned API endpoints (`/api/v1/...`)

### API-Ready Principles
- Keep service logic framework-agnostic
- Return JSON from API routes and HTML from web routes
- Reuse validation/business service functions in both channels

## 7) Initial MySQL Entity Outline
- `users`
- `roles`
- `products`
- `product_media` (images/videos)
- `product_variants`
- `inventory_movements`
- `carts`
- `cart_items`
- `orders`
- `order_items`
- `addresses`
- `payments`
- `discount_codes`
- `discount_redemptions`
- `reviews`
- `review_votes`
- `shipping_rates`

## 8) UX / IA Notes
Primary pages:
- `/` Home
- `/shop`
- `/product/{slug}`
- `/cart`
- `/checkout`
- `/login`, `/register`
- `/account`
- `/admin` (+ subpages)

Navigation tone: premium visual language with clear affordability cues (promotions, bundles, social proof).

## 9) Security & Compliance Baseline
- Password hashing with `password_hash`
- Prepared statements for SQL
- CSRF protection on state-changing forms
- Upload validation and MIME restrictions
- Admin audit trail for inventory adjustments

## 10) Milestone Plan
1. Design sign-off (wireframe + spec)
2. Project skeleton and environment bootstrap
3. Auth + roles + admin shell
4. Product + variant + media + inventory
5. Cart + checkout + payments (PayPal + bank transfer flow)
6. Shipping rules (AU + NG logic)
7. Reviews + discount codes
8. Reporting + low-stock alerts
9. Hardening + test pass + launch checklist

