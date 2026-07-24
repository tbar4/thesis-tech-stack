-- Latest element set per object from the shippable celestrak snapshot tier.
-- Embargoed spacetrack data is deliberately absent from gold.
with all_sets as (
    select *
    from read_parquet(
        '{{ env_var("SDA_DATA_HOME") }}/data/snapshots/celestrak/element_sets/**/*.parquet',
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
