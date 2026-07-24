
  
    
    

    create  table
      "sda"."main"."latest_element_sets__dbt_tmp"
  
    as (
      -- Latest element set per object from the shippable celestrak snapshot tier.
-- Embargoed spacetrack data is deliberately absent from gold.
with all_sets as (
    select *
    from read_parquet(
        '/tmp/pytest-of-root/pytest-14/test_dbt_build_produces_gold0/data/snapshots/celestrak/element_sets/**/*.parquet',
        union_by_name = true
    )
)
select * exclude (rn)
from (
    select *,
           row_number() over (partition by norad_cat_id order by fetch_time desc) as rn
    from all_sets
)
where rn = 1
    );
  
  