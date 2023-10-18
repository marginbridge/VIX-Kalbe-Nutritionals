--Berapa rata-rata umur customer jika dilihat dari marital statusnya ?
select "Marital Status", avg(age)
from "Customer" c 
group by "Marital Status" 

--Berapa rata-rata umur customer jika dilihat dari gender nya ?
select gender, avg(age)
from "Customer" c 
group by gender 

--Tentukan nama store dengan total quantity terbanyak!
select s.storename, count(t.qty)
from "Store" s
join "Transaction" t on s.storeid = t.storeid
group by s.storename 

--Tentukan nama produk terlaris dengan total amount terbanyak!
select p."Product Name" , sum(t.totalamount) as "Total Amount"
from "Product" p 
join "Transaction" t on p.productid = t.productid 
group by p."Product Name"
order by "Total Amount" desc 