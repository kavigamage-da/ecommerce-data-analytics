# Data Dictionary

## Customer Profiles

| Column Name | Data Type | Description |
|-------------|----------|-------------|
| customer_id | int64 | Customer id |
| name | str | Name |
| gender | str | Gender |
| age | int64 | Age |
| country | str | Country |
| signup_date | str | Signup date |
| customer_tier | str | Customer tier |
| months_since_signup | int64 | Months since signup |
| churned | bool | Churned |
| CLV | float64 | Clv |
| next_purchase_prob | float64 | Next purchase prob |

## Purchase History

| Column Name | Data Type | Description |
|-------------|----------|-------------|
| order_id | int64 | Order id |
| customer_id | int64 | Customer id |
| order_date | str | Order date |
| order_amount | float64 | Order amount |
| product_category | str | Product category |
| discount_used | bool | Discount used |
| discount_value | float64 | Discount value |

## Engagement Behavior

| Column Name | Data Type | Description |
|-------------|----------|-------------|
| customer_id | int64 | Customer id |
| weekly_visits | int64 | Weekly visits |
| session_time_minutes | float64 | Session time minutes |
| page_views | float64 | Page views |
| app_opens | int64 | App opens |

## Marketing Promotions

| Column Name | Data Type | Description |
|-------------|----------|-------------|
| customer_id | int64 | Customer id |
| campaign_name | str | Campaign name |
| discount | float64 | Discount |
| responded | bool | Responded |
| additional_revenue | float64 | Additional revenue |

