-- Deduped article metadata: the queryable face of the 8.1 RAG text branch.
select * exclude (rn)
from (
    select *,
           row_number() over (partition by id order by fetch_time desc) as rn
    from read_parquet(
        '{{ env_var("SDA_DATA_HOME") }}/data/snapshots/spaceflightnews/articles/**/*.parquet',
        union_by_name = true
    )
)
where rn = 1
