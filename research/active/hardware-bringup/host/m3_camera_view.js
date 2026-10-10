'use strict';

// One streaming HTTP response; one decode in flight and one replaceable latest frame.
class CameraMultipart {
 constructor(onFrame){this.onFrame=onFrame;this.buffer=new Uint8Array();this.header=null;this.length=0}
 push(chunk){
  if(this.buffer.length+chunk.length>2*1024*1024)throw Error('画面数据过大');
  const joined=new Uint8Array(this.buffer.length+chunk.length);joined.set(this.buffer);joined.set(chunk,this.buffer.length);this.buffer=joined;
  while(true){
   if(!this.header){
    let end=-1;for(let i=0;i+3<this.buffer.length;i++)if(this.buffer[i]===13&&this.buffer[i+1]===10&&this.buffer[i+2]===13&&this.buffer[i+3]===10){end=i;break}
    if(end<0){if(this.buffer.length>4096)throw Error('画面头无效');return}
    const lines=new TextDecoder().decode(this.buffer.subarray(0,end)).trim().split('\r\n');if(lines.shift()!=='--BAFRAME')throw Error('画面边界无效');
    const header={};for(const line of lines){const i=line.indexOf(':');if(i<1)throw Error('画面头无效');const key=line.slice(0,i).toLowerCase();if(key in header)throw Error('画面头重复');header[key]=line.slice(i+1).trim()}
    const length=Number(header['content-length']);if(header['content-type']!=='image/jpeg'||!Number.isInteger(length)||length<1||length>1024*1024||!/^\d+$/.test(header['x-frame-sequence']||'')||!/^[\w-]{1,128}$/.test(header['x-sequence-id']||''))throw Error('画面头无效');
    const age=Number(header['x-host-age-ms']),sent=Number(header['x-host-sent-unix-ms']);if(!('x-host-age-ms' in header)||!('x-host-sent-unix-ms' in header)||!Number.isFinite(age)||age<0||!Number.isFinite(sent)||sent<=0)throw Error('画面时龄无效');
    this.header=header;this.length=length;this.buffer=this.buffer.slice(end+4);
   }
   if(this.buffer.length<this.length+2)return;
   if(this.buffer[this.length]!==13||this.buffer[this.length+1]!==10)throw Error('画面末尾无效');
   const frame={jpeg:this.buffer.slice(0,this.length),header:this.header,received:performance.now()};this.buffer=this.buffer.slice(this.length+2);this.header=null;this.onFrame(frame);
  }
 }
}

class CameraView {
 constructor(){this.controller=null;this.epoch=0;this.retry=null;this.running=false;this.decoding=false;this.pending=null;this.key=null;this.count=0;this.lastPaint=null;this.times=[];this.frame=null;this.connections=0;this.health=null;
  document.addEventListener('visibilitychange',()=>{if(document.hidden)this.pause('页面暂停显示');else this.schedule(0)});
  window.addEventListener('pagehide',()=>this.pause('画面已结束',true));
 }
 hide(message){el('cameraImage').classList.add('hidden');el('cameraWait').classList.remove('hidden');el('cameraWait').textContent=message;this.lastPaint=null;this.times=[];this.key=null;this.frame=null;this.info()}
 info(){const fps=this.times.length>1?(this.times.length-1)*1000/(this.times.at(-1)-this.times[0]):null;el('cameraInfo').textContent=fps===null?'等待实时画面':'画面每秒更新约 '+label(fps,0)+' 次';el('cameraDetail').textContent='相机接收每秒 '+label(latest?.camera?.fps,1)+' 帧 · 页面更新每秒 '+label(fps,1)+' 次'+(this.frame===null?'':' · 帧序号 '+this.frame);const canvas=el('cameraImage');canvas.dataset.presentedFrames=String(this.count);canvas.dataset.presentedFps=fps===null?'':String(fps);canvas.dataset.streamConnections=String(this.connections)}
 schedule(delay){clearTimeout(this.retry);this.retry=null;if(!ended&&!document.hidden&&!latest?.wifi?.busy)this.retry=setTimeout(()=>{this.retry=null;this.connect()},delay)}
 pause(message,final=false){this.epoch++;this.pending=null;clearTimeout(this.retry);if(this.controller)this.controller.abort();this.hide(message);if(final)clearInterval(this.health);else if(!this.running)this.schedule(500)}
 status(s){if(s.wifi?.busy){if(this.controller||this.lastPaint!==null)this.pause('正在切换相机 Wi-Fi');el('cameraState').textContent='正在切换相机 Wi-Fi';return}if(this.lastPaint!==null&&performance.now()-this.lastPaint<1500){el('cameraState').textContent='实时画面';this.info()}else{el('cameraState').textContent=s.camera?.status||'等待相机';if(this.lastPaint!==null)this.pause('等待新画面');else this.info()}if(!this.running&&this.retry===null)this.schedule(0)}
 start(){this.health=setInterval(()=>{if(this.lastPaint!==null&&performance.now()-this.lastPaint>=1500)this.pause('画面中断，正在重连')},250);this.schedule(0)}
 async connect(){if(this.running||ended||document.hidden)return;if(!latest?.camera?.available||latest?.wifi?.busy){this.schedule(500);return}this.running=true;const epoch=this.epoch,controller=new AbortController();this.controller=controller;let reader=null,watchdog=null;
  try{const response=await fetch('/api/camera.mjpeg',{cache:'no-store',signal:controller.signal});if(!response.ok||!response.headers.get('Content-Type')?.includes('boundary=BAFRAME'))throw Error('等待相机');reader=response.body.getReader();this.connections++;let lastData=performance.now();watchdog=setInterval(()=>{if(performance.now()-lastData>1500)controller.abort()},250);
   const parser=new CameraMultipart(frame=>{if(epoch!==this.epoch)return;this.pending={...frame,epoch};this.decode()});
   while(epoch===this.epoch&&!ended&&!document.hidden){const {value,done}=await reader.read();if(done)throw Error('画面连接中断');lastData=performance.now();parser.push(value)}
  }catch(e){if(epoch===this.epoch&&!ended&&!document.hidden)this.pause('等待新画面')}
  finally{clearInterval(watchdog);controller.abort();if(reader){try{await reader.cancel()}catch(e){}reader.releaseLock()}if(epoch===this.epoch)this.pending=null;this.controller=null;this.running=false;this.schedule(500)}
 }
 async decode(){if(this.decoding)return;this.decoding=true;
  try{while(this.pending){const frame=this.pending;this.pending=null;let bitmap=null;try{const h=frame.header,key=h['x-sequence-id']+':'+h['x-frame-sequence'];if(key===this.key)continue;bitmap=await createImageBitmap(new Blob([frame.jpeg],{type:'image/jpeg'}));if(ended||document.hidden||frame.epoch!==this.epoch||latest?.wifi?.busy)continue;
     const age=Number(h['x-host-age-ms'])+Math.max(0,Date.now()-Number(h['x-host-sent-unix-ms']));if(age>=1500){this.pause('画面过期，正在重连');continue}
     const canvas=el('cameraImage');canvas.getContext('2d').drawImage(bitmap,0,0,canvas.width,canvas.height);this.key=key;this.frame=h['x-frame-sequence'];this.lastPaint=performance.now();this.count++;this.times.push(this.lastPaint);if(this.times.length>90)this.times.shift();canvas.dataset.lastPaintMs=String(this.lastPaint);canvas.dataset.frameKey=key;canvas.classList.remove('hidden');el('cameraWait').classList.add('hidden');el('cameraState').textContent='实时画面';this.info();
    }finally{if(bitmap)bitmap.close()}}
  }catch(e){this.pause('画面解码失败，正在重连')}finally{this.decoding=false}
 }
}
const cameraView=new CameraView();
function camera(s){cameraView.status(s)}
function cameraPause(message){cameraView.pause(message,ended)}
