
  
    
    

    create  table
      "sda"."main"."latest_articles__dbt_tmp"
  
    as (
      -- Deduped article metadata: the queryable face of the 8.1 RAG text branch.
select * exclude (rn)
from (
    select *,
           row_number() over (partition by id order by fetch_time desc) as rn
    from read_parquet(
        '/tmp/pytest-of-root/pytest-14/test_dbt_build_produces_gold0/data/snapshots/spaceflightnews/articles/**/*.parquet',
        union_by_name = true
    )
)
where rn = 1
    );
  
  