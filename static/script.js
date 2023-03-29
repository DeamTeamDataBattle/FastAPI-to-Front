function toggle_hide(id) {
  let ele = document.getElementById(id);
  if (ele.classList.contains("hidden")) {
    ele.classList.remove("hidden")
  }else{
    ele.classList.add("hidden")
  }
}

function update_progress(id, value) {
  document.getElementById(id).value = value;
}

function reset_upload() {
  document.getElementById("upload-info").innerText = "";
  //document.getElementById("upload-info").classList.add("hidden");
  document.getElementById("upload-progress-bar").classList.add("hidden");
  document.getElementById("upload-error").classList.add("hidden");
  
  // show pre info about doc size and estimate time
  let pdf = document.getElementById("pdf-upload").files[0];
  console.log(pdf);
  let size = Math.round(pdf["size"]/1e4)/100;
  let est_time = Math.round(size * 100 / 7.75) / 100;
  //toggle_hide("upload-info");
  document.getElementById("upload-info").innerText = "Doc size: " + size + "MB\nEstimate time: " + est_time + "s";
}

async function get_notifications() {
  let res = await fetch("/notif", {method: "GET"});
  res = await res.json();
  let notif_text = res["notif"];
  document.getElementById("upload-info").innerText = notif_text;
}

function gotopie(data) {
  keys = ["sand", "clay", "lime", "shale", "salt", "silt", "chert"];
  values = Array(keys.length).fill(0);
  let total = 0;
  comp = data["data"];
  for (var key in comp){
    for (let i=0; i<keys.length; i++){
      if (key.includes(keys[i])){
        values[i] += comp[key];
        total += comp[key];
      }
    }
  }
  let sand = Math.round(100*values[0]/total);
  let clay = Math.round(100*values[1]/total)
  window.location.href = "/pie?name="+data["file"]+"&sandstone="+sand+"&clay="+clay;
}

async function upload_pdf() {
  let pdf = document.getElementById("pdf-upload").files[0];
  if (typeof pdf === 'undefined') {
    alert("please select a document first");
  } else {
    //toggle_hide("upload-progress-bar");
    //update_progress("upload-progress-bar", 0);

    // send request
    let body = new FormData();
    body.append("file", pdf);
    document.getElementById("upload-info").innerText = "start";
    let notifID = setInterval(get_notifications, 200);
    let res = await fetch("/upload-pdf", {method: "POST", body: body});
    let data = await res.json().then(clearInterval(notifID));
    console.log(data);
    // update page with html
    //update_progress("upload-progress-bar", 100);
    if ("error" in data) {
      toggle_hide("upload-error");
      document.getElementById("upload-error").innerText = data["error"];
    } else {
      document.getElementById("upload-info").innerText = data["info"];
      setTimeout(() => {
        gotopie(data);
      },
      1500);
    }
    document.getElementById("pdf-upload").value = "";
  }
}
