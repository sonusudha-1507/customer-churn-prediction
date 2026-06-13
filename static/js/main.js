document
.getElementById("prediction-form")
.addEventListener(
"submit",

async function(e){

e.preventDefault();


let data=
new FormData(this);


let response=
await fetch(
"/predict",
{
method:"POST",
body:data
}
);


let result=
await response.json();


if(result.success){


document
.getElementById("probability")
.innerText
=
result.probability+"%";


document
.getElementById("risk")
.innerText
=
result.churn_risk+" Risk";


let list=
document.getElementById("drivers");


list.innerHTML="";


result.drivers.forEach(

x=>{

list.innerHTML+=

`
<li>
${x.feature}
Impact: ${x.impact}
</li>
`

}

);


}



}

);