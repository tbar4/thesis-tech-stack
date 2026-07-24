
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select neo_reference_id
from "sda"."main"."neo_approaches"
where neo_reference_id is null



  
  
      
    ) dbt_internal_test