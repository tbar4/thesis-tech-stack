
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select id
from "sda"."main"."latest_articles"
where id is null



  
  
      
    ) dbt_internal_test