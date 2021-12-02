function Enter_Check(){
  // 엔터키의 코드는 13입니다.
  if(event.keyCode == 13){
    createDiv();  // 실행할 이벤트
  }
}
function test() {
  document.body.scrollTop = document.body.scrollHeight;
}
function createDiv() {
if (document.getElementsByClassName('form-control')[0].value === "")
  return;
  // 1. <div> element 만들기
  var qlDiv = document.createElement('div');
  var qrDiv = document.createElement('div');
  var qqDiv = document.createElement('div');
  var qqqDiv = document.createElement('div');
  
  qlDiv.className = "chat-message-left pb-4";
  qrDiv.className = "chat-message-right pb-4";
  qqDiv.className = "flex-shrink-1 bg-light rounded py-2 px-3 ml-3";
  qqqDiv.className = "font-weight-bold mb-1";
  // 2. <div>에 들어갈 text node 만들기
  
  var name = '나';
  var qqqnewText = document.createTextNode(name);
  var question = document.getElementsByClassName('form-control')[0].value;
  console.log(question);
  //var question = document.getElementById("inputbox").value;
  var qqnewText = document.createTextNode(question);
  
  // 3. <div>에 text node 붙이기
  qqqDiv.appendChild(qqqnewText);
  qqDiv.appendChild(qqqDiv)
  qqDiv.appendChild(qqnewText);
  
  qrDiv.appendChild(qqDiv);
  

  

  var t = document.getElementsByClassName("chat-messages p-4")[0];
  t.appendChild(qrDiv);

  fetch("/api", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        "sentence": question
      }),
    }) 
    .then((response) => response.json())
    .then((data) => {
      
      //var newText = document.createTextNode(answer);
      var qlDiv = document.createElement('div');
      var qqDiv = document.createElement('div');
      var qqqDiv = document.createElement('div');
          
      qlDiv.className = "chat-message-left pb-4";
      qqDiv.className = "flex-shrink-1 bg-light rounded py-2 px-3 ml-3";
      qqqDiv.className = "font-weight-bold mb-1";
      
      var name = '도라미';
      var img = document.createElement("img");
      img.src = "../static/css/image/dorami.png";
      img.width = 50;
      img.height = 50;
      var qqqnewText = document.createTextNode(name);
      var answer = data["pred"];
      console.log(answer);
      var answerText = document.createTextNode(answer);
      
      // 3. <div>에 text node 붙이기
      qqqDiv.appendChild(img);
      qqqDiv.appendChild(qqqnewText);
      qqDiv.appendChild(qqqDiv)
      
      qqDiv.appendChild(answerText);

      qlDiv.appendChild(qqDiv);
      t.appendChild(qlDiv);

      var objDiv = document.getElementById("mydiv"); 
      objDiv.scrollTop = objDiv.scrollHeight;

      var emotion_cnt = Cookies.get(data['emotion']);
      console.log(data['emotion'])
      emotion_cnt = emotion_cnt * data['emotion_cnt'] + data['emotion_cnt']
      console.log(emotion_cnt)
      
      Cookies.set(data['emotion'], emotion_cnt)
      });

    document.getElementsByClassName('form-control')[0].value='';
} 
