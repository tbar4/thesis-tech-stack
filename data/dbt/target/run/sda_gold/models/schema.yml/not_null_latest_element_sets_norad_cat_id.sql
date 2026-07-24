
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select norad_cat_id
from "sda"."main"."latest_element_sets"
where norad_cat_id is null



  
  
      
    ) dbt_internal_test