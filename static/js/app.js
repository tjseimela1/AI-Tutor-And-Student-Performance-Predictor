(function(){
    const menu = document.getElementById('menuButton');
    const sidebar = document.getElementById('sidebar');
    if(menu && sidebar){
      menu.addEventListener('click',()=>{
        const opened = sidebar.classList.toggle('open');
        menu.setAttribute('aria-expanded',String(opened));
      });
      document.addEventListener('click',(e)=>{
        if(sidebar.classList.contains('open') && !sidebar.contains(e.target) && !menu.contains(e.target)){
          sidebar.classList.remove('open');menu.setAttribute('aria-expanded','false');
        }
      });
    }
    const stage = document.getElementById('schoolStage');
    const grade = document.getElementById('schoolGrade');
    const schoolMap = document.getElementById('stageGradeMap');
    if(stage && grade && schoolMap){
      try {
        const mappings = JSON.parse(schoolMap.textContent);
        function updateGrades(){
          const current = grade.value;
          grade.replaceChildren();
          (mappings[stage.value] || []).forEach(code => {
            const option = document.createElement('option');
            option.value = code; option.textContent = 'Grade ' + Number(code.slice(2));
            grade.appendChild(option);
          });
          if([...grade.options].some(x => x.value === current)){grade.value = current;}
          else if([...grade.options].some(x=>x.value === 'G-07')){grade.value='G-07';}
        }
        stage.addEventListener('change',updateGrades);updateGrades();
      } catch(e) { /* display safe default if malformed mapping */ }
    }
    const form = document.getElementById('chatForm');
    if(!form) return;
    const input = document.getElementById('chatInput');
    const feed = document.getElementById('chatMessages');
    function bubble(message, isUser, topic, resource){
      const box=document.createElement('div');
      box.className='chat-bubble '+(isUser?'user-bubble':'bot-bubble');
      if(!isUser){const heading=document.createElement('strong');heading.textContent='✳ Study Assistant'+(topic?' · '+topic:'');box.appendChild(heading);}
      const p=document.createElement('p');p.textContent=message;box.appendChild(p);
      if(resource && /^https:\/\//.test(resource)){
        const p2=document.createElement('p');
        const link=document.createElement('a');link.href=resource;link.target='_blank';link.rel='noopener noreferrer';link.textContent='Learn more from a study resource ↗';
        p2.appendChild(link);box.appendChild(p2);
      }
      feed.appendChild(box);feed.scrollTop=feed.scrollHeight;
    }
    async function ask(question){
      question=String(question||'').trim().slice(0,500);
      if(!question) return;
      bubble(question,true);input.value='';
      const button=form.querySelector('button');button.disabled=true;
      try{
        const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})});
        if(!response.ok)throw new Error('Request failed');
        const result=await response.json();bubble(result.answer,false,result.topic,result.resource);
      }catch(err){bubble('Sorry, I could not retrieve an answer right now. Please try again.',false);}
      finally{button.disabled=false;input.focus();}
    }
    form.addEventListener('submit',(e)=>{e.preventDefault();ask(input.value);});
    document.querySelectorAll('[data-question]').forEach(button=>button.addEventListener('click',()=>ask(button.dataset.question)));
  })();
  