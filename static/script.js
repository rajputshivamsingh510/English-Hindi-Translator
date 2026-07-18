const translateBtn=document.getElementById("translateBtn");

const english=document.getElementById("english");

const hindi=document.getElementById("hindi");

const loader=document.getElementById("loader");

translateBtn.onclick=async()=>{

if(english.value.trim()===""){

alert("Please enter text.");

return;

}

loader.classList.remove("hidden");

hindi.value="";

const response=await fetch("/translate",{

method:"POST",

headers:{

"Content-Type":"application/json"

},

body:JSON.stringify({

text:english.value

})

});

const data=await response.json();

loader.classList.add("hidden");

hindi.value=data.translation;

};

document.getElementById("copyBtn").onclick=()=>{

navigator.clipboard.writeText(hindi.value);

alert("Copied!");

};

document.getElementById("clearBtn").onclick=()=>{

english.value="";

hindi.value="";

};
