-- Deduped launches still in the future (net = no-earlier-than).
with all_launches as (
    select *
    from read_parquet(
        '/tmp/pytest-of-root/pytest-14/test_dbt_build_produces_gold0/data/snapshots/spacedevs/launches/**/*.parquet',
        union_by_name = true
    )
),
latest as (
    select * exclude (rn)
    from (
        select *,
               row_number() over (partition by id order by fetch_time desc) as rn
        from all_launches
    )
    where rn = 1
)
select *
from latest
where try_cast(net as timestamptz) >= now()