-- Deduped NASA NeoWs close approaches (derived facts; raw dumps stay embargoed).
select * exclude (rn)
from (
    select *,
           row_number() over (
               partition by neo_reference_id, close_approach_date
               order by fetch_time desc
           ) as rn
    from read_parquet(
        '/tmp/pytest-of-root/pytest-14/test_dbt_build_produces_gold0/data/snapshots/nasa/neows/**/*.parquet',
        union_by_name = true
    )
)
where rn = 1