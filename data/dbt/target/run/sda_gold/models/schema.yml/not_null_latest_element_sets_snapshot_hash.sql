
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select snapshot_hash
from "sda"."main"."latest_element_sets"
where snapshot_hash is null



  
  
      
    ) dbt_internal_test