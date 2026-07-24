
    
    

select
    id as unique_field,
    count(*) as n_records

from "sda"."main"."latest_articles"
where id is not null
group by id
having count(*) > 1


