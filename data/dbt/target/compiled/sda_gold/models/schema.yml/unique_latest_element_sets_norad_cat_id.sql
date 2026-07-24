
    
    

select
    norad_cat_id as unique_field,
    count(*) as n_records

from "sda"."main"."latest_element_sets"
where norad_cat_id is not null
group by norad_cat_id
having count(*) > 1


