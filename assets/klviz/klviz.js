/* Dependency-free SVG charts. Analytical examples and local, frozen experiment data.
   Styles use CSS variables, so changing color scheme needs no redraw. */
(() => {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  let dismissFloatingTooltip;
  // A viewport-positioned card must not stay behind when the page moves.
  window.addEventListener('scroll',()=>dismissFloatingTooltip?.(),{capture:true,passive:true});
  window.addEventListener('resize',()=>dismissFloatingTooltip?.());
  const colors = {k1: 'var(--kl-blue)', k3: 'var(--kl-orange)', LLM: 'var(--kl-purple)', PPO: 'var(--kl-green)'};
  const sum = a => a.reduce((x, y) => x + y, 0);
  const mean = a => sum(a) / a.length;
  const quantile = (a, p) => {
    const s = [...a].sort((x, y) => x - y), i = (s.length - 1) * p, j = Math.floor(i);
    return s[j] + (s[Math.ceil(i)] - s[j]) * (i - j);
  };
  const num = n => {
    if (n === 0) return '0';
    if (Math.abs(n) < .001 || Math.abs(n) >= 10000) return n.toExponential(2).replace('e+', 'e').replace('e-', 'e−');
    return Number(n.toPrecision(4)).toLocaleString('en-US', {maximumFractionDigits: 6});
  };
  const power = n => '10' + String(n).split('').map(c => ({'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'}[c])).join('');
  const svgEl = (name, attrs = {}, text) => {
    const e = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    if (text !== undefined) e.textContent = text;
    return e;
  };
  function appendMathText(parent, text) {
    String(text).split(/(\$[^$]+\$|\b[kp][123]\b|[kp][₁₂₃])/).forEach(part => {
      const explicit = part.startsWith('$') && part.endsWith('$');
      if ((explicit || /^[kp][123₁₂₃]$/.test(part)) && window.katex) {
        const span=document.createElement('span');span.className='kl-inline-label';
        const sub={'₁':'1','₂':'2','₃':'3'}[part[1]] || part[1];
        const latex=explicit ? part.slice(1,-1) : `${part[0]}_${sub}`;
        window.katex.render(latex,span,{displayMode:false,throwOnError:false});
        parent.append(span);
      } else parent.append(document.createTextNode(part));
    });
  }
  const el = (name, className, parent, text) => {
    const e = document.createElement(name);
    if (className) e.className = className;
    if (text !== undefined) appendMathText(e, text);
    if (parent) parent.append(e);
    return e;
  };
  function tabs(parent, options, initial, onChange) {
    const group = el('div', 'kl-tabs', parent);
    options.forEach(([value, label]) => {
      const button = el('button', '', group, label);
      button.type = 'button'; button.setAttribute('aria-pressed', String(value === initial));
      button.addEventListener('click', () => {
        [...group.children].forEach(b => b.setAttribute('aria-pressed', String(b === button)));
        onChange(value);
      });
    });
    return group;
  }
  function legend(parent, series, reference) {
    parent.replaceChildren();
    [...series, ...(reference ? [{label: reference.label, color: 'var(--kl-muted)', dashed: true}] : [])].forEach(s => {
      const item = el('span', '', parent);
      const sw = el('i', s.dashed ? 'dashed' : '', item);
      sw.style.setProperty('--series', s.color);
      sw.setAttribute('aria-hidden', 'true');
      appendMathText(item,s.label);
    });
  }
  function readouts(parent, values) {
    parent.replaceChildren();
    values.forEach(([label, value, color]) => {
      const item = el('div', 'kl-readout', parent);
      el('small', '', item, label);
      const strong = el('strong', '', item, value);
      if (color) strong.style.color = color;
    });
  }
  function niceTicks(domain, log, count = 5) {
    if (log) {
      const lo = Math.ceil(Math.log10(domain[0])), hi = Math.floor(Math.log10(domain[1]));
      const stride = Math.max(1, Math.ceil((hi - lo) / count));
      const ticks = [];
      for (let i = Math.ceil(lo / stride) * stride; i <= hi; i += stride) ticks.push(10 ** i);
      return ticks;
    }
    const raw = (domain[1] - domain[0]) / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].find(x => x * mag >= raw) * mag;
    const ticks = [];
    for (let x = Math.ceil(domain[0] / step) * step; x <= domain[1] + step * 1e-8; x += step) ticks.push(x);
    return ticks;
  }
  function floatingTooltipPosition({cx,cy,width,height,vw,vh,plots,previous}) {
    const gap=14, edge=10;
    const candidates=[['above-right',true,true],['above-left',false,true],['below-right',true,false],['below-left',false,false]].map(([corner,right,above])=>{
      const left=Math.max(edge,Math.min(vw-edge-width,right?cx+gap:cx-gap-width));
      const top=Math.max(edge,Math.min(vh-edge-height,above?cy-gap-height:cy+gap));
      const overlap=plots.reduce((area,plot)=>area+
        Math.max(0,Math.min(left+width,plot.right)-Math.max(left,plot.left))*
        Math.max(0,Math.min(top+height,plot.bottom)-Math.max(top,plot.top)),0);
      const coversPointer=left<cx+8 && left+width>cx-8 && top<cy+8 && top+height>cy-8;
      // A small preference for the current corner prevents jitter at a tie.
      return {left,top,corner,coversPointer,score:overlap+(previous && corner!==previous?120:0)};
    });
    const clear=candidates.filter(p=>!p.coversPointer);
    return (clear.length?clear:candidates).reduce((best,p)=>p.score<best.score?p:best);
  }
  function lineChart(parent, cfg) {
    parent.replaceChildren();
    const w = Math.max(280, parent.clientWidth), h = w < 480 ? 300 : 342;
    const m = {l: w < 480 ? 57 : 70, r: 17, t: 16, b: 53};
    const pw = w - m.l - m.r, ph = h - m.t - m.b;
    const tx = x => cfg.logX ? Math.log10(x) : x;
    const ty = y => cfg.logY ? Math.log10(y) : y;
    const xd = cfg.xDomain || [Math.min(...cfg.series.flatMap(s => s.data.map(p => p.x))), Math.max(...cfg.series.flatMap(s => s.data.map(p => p.x)))];
    const ys = cfg.series.flatMap(s => s.data.filter(p => p.y != null).flatMap(p => [p.y, p.lo ?? p.y, p.hi ?? p.y]));
    if (cfg.reference) ys.push(cfg.reference.y);
    const bounds = [Math.min(...ys.map(ty)), Math.max(...ys.map(ty))];
    const pad = (bounds[1] - bounds[0]) * .10 || .1;
    const yd = cfg.yDomain || (cfg.logY ? [10 ** (bounds[0] - pad), 10 ** (bounds[1] + pad)] : [Math.min(0, bounds[0] - pad), bounds[1] + pad]);
    const x = v => m.l + (tx(v) - tx(xd[0])) / (tx(xd[1]) - tx(xd[0])) * pw;
    const y = v => m.t + ph - (ty(v) - ty(yd[0])) / (ty(yd[1]) - ty(yd[0])) * ph;
    const svg = svgEl('svg', {viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': cfg.description});
    parent.append(svg); parent.tabIndex = 0;
    parent.setAttribute('aria-label', cfg.description + ' Use left and right arrow keys to inspect values.');
    const tooltip = el('div', cfg.floatingTooltip ? 'kl-tooltip kl-tooltip-floating' : 'kl-tooltip', parent); tooltip.hidden = true;
    tooltip.setAttribute('role', 'status');
    const add = (tag, attrs, text) => {const e = svgEl(tag, attrs, text); svg.append(e); return e;};
    const yTicks = cfg.yTicks || niceTicks(yd, cfg.logY);
    yTicks.forEach(v => {
      add('line', {x1: m.l, x2: w-m.r, y1: y(v), y2: y(v), class: 'kl-grid'});
      add('text', {x: m.l-11, y: y(v)+4, 'text-anchor':'end', class:'kl-tick'}, cfg.percent ? `${Math.round(v*100)}%` : cfg.logY ? power(Math.round(Math.log10(v))) : num(v));
    });
    (cfg.xTicks || niceTicks(xd, cfg.logX, w < 480 ? 3 : 5)).forEach(v => {
      add('line', {x1:x(v), x2:x(v), y1:h-m.b, y2:h-m.b+5, class:'kl-axis'});
      add('text', {x:x(v), y:h-m.b+21, 'text-anchor':'middle', class:'kl-tick'}, cfg.xFormat ? cfg.xFormat(v) : cfg.logX ? power(Math.round(Math.log10(v))) : num(v));
    });
    add('line', {x1:m.l, x2:w-m.r, y1:h-m.b, y2:h-m.b, class:'kl-axis'});
    add('text', {x:m.l+pw/2, y:h-5, 'text-anchor':'middle', class:'kl-axis-label'}, cfg.xLabel);
    add('text', {transform:`translate(13 ${m.t+ph/2}) rotate(-90)`, 'text-anchor':'middle', class:'kl-axis-label'}, cfg.yLabel);
    if (cfg.reference) add('line', {x1:m.l,x2:w-m.r,y1:y(cfg.reference.y),y2:y(cfg.reference.y),class:'kl-guide'});
    const pointsPath = (points, value) => points.map((p,i) => `${i?'L':'M'}${x(p.x)},${y(p[value])}`).join(' ');
    cfg.series.forEach(s => {
      // Preserve gaps rather than connecting bins with insufficient observations.
      const segments = [[]];
      s.data.forEach(p => {if (p.y == null) segments.push([]); else segments.at(-1).push(p);});
      segments.filter(a => a.length).forEach(a => {
        if (a.every(p => p.lo != null && p.hi != null)) {
          add('path', {d:pointsPath(a,'lo') + ' ' + pointsPath([...a].reverse(),'hi').replace(/^M/,'L') + ' Z', class:'kl-band', style:`--series:${s.color}`});
        }
        add('path', {d:pointsPath(a,'y'), class:'kl-series', style:`--series:${s.color}`, ...(s.dashed ? {'stroke-dasharray':'6 5'} : {})});
        if (cfg.dots !== false) a.forEach(p => add('circle', {cx:x(p.x),cy:y(p.y),r:3.5,class:'kl-dot',style:`--series:${s.color}`}));
      });
    });
    const guide = add('line', {x1:0,x2:0,y1:m.t,y2:h-m.b,class:'kl-guide',visibility:'hidden'});
    const hit = add('rect', {x:m.l,y:m.t,width:pw,height:ph,class:'kl-hit'});
    const xs = [...new Set(cfg.series.flatMap(s => s.data.filter(p => p.y != null).map(p => p.x)))].sort((a,b)=>a-b);
    let index = 0, tooltipCorner = null;
    function hideTooltip() {
      tooltip.hidden=true;guide.setAttribute('visibility','hidden');
      tooltipCorner=null;
      if(dismissFloatingTooltip===hideTooltip) dismissFloatingTooltip=null;
    }
    function inspect(i, fixed = false, pointer = null) {
      index = Math.max(0, Math.min(xs.length-1, i));
      const value = xs[index];
      guide.setAttribute('x1', x(value)); guide.setAttribute('x2', x(value)); guide.setAttribute('visibility','visible');
      tooltip.replaceChildren();
      if (cfg.renderTooltip) {
        cfg.renderTooltip(tooltip, value);
      } else {
        el('b','',tooltip, `${cfg.xTip || cfg.xLabel}: ${num(value)}`);
        cfg.series.forEach(s => {
          const p = s.data.find(p => p.x === value && p.y != null);
          if (!p) return;
          const row = el('div','',tooltip, `${s.label}: ${cfg.percent ? (p.y*100).toFixed(2)+'%' : num(p.y)}`);
          row.style.color = s.color;
        });
      }
      tooltip.hidden = false;
      const px = pointer ? pointer.x : x(value), py = pointer ? pointer.y : 8;
      if(cfg.floatingTooltip) {
        if(dismissFloatingTooltip && dismissFloatingTooltip!==hideTooltip) dismissFloatingTooltip();
        dismissFloatingTooltip=hideTooltip;
        const rect=svg.getBoundingClientRect(), cx=rect.left+px*rect.width/w, cy=rect.top+py*rect.height/h;
        // Protect both plots in the pair, rather than moving the card from
        // the active plot onto its neighbor. Cards may leave the figure.
        const plots=[...parent.closest('.kl-gradient-grid').querySelectorAll('.kl-hit')].map(p=>p.getBoundingClientRect());
        const position=floatingTooltipPosition({cx,cy,width:tooltip.offsetWidth,height:tooltip.offsetHeight,vw:document.documentElement.clientWidth,vh:window.innerHeight,plots,previous:tooltipCorner});
        tooltipCorner=position.corner;
        tooltip.style.left=`${position.left}px`;
        tooltip.style.top=`${position.top}px`;
      } else {
        const right = px + 16;
        const left = right + tooltip.offsetWidth > w ? px - tooltip.offsetWidth - 16 : right;
        tooltip.style.left = `${Math.max(0, Math.min(w-tooltip.offsetWidth, left))}px`;
        tooltip.style.top = `${Math.max(0, Math.min(h-tooltip.offsetHeight, py+16))}px`;
      }
      if (fixed) tooltip.setAttribute('aria-live','polite'); else tooltip.removeAttribute('aria-live');
    }
    hit.addEventListener('pointermove', e => {
      const rect = svg.getBoundingClientRect(), pos = (e.clientX-rect.left)*w/rect.width;
      inspect(xs.reduce((best,v,i)=>Math.abs(x(v)-pos)<Math.abs(x(xs[best])-pos)?i:best,0), false, {x:pos,y:(e.clientY-rect.top)*h/rect.height});
    });
    parent.onpointerleave = hideTooltip;
    if(cfg.floatingTooltip) parent.onblur=hideTooltip;
    parent.onkeydown = e => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {e.preventDefault();inspect(index+(e.key==='ArrowRight'?1:-1),true);}
      if (e.key === 'Escape') hideTooltip();
    };
    if (cfg.selectedX != null) {
      const v=cfg.selectedX;
      guide.setAttribute('x1',x(v));guide.setAttribute('x2',x(v));guide.setAttribute('visibility','visible');
      cfg.series.forEach(s => {
        const p=s.data.reduce((best,p)=>Math.abs(p.x-v)<Math.abs(best.x-v)?p:best);
        add('circle',{cx:x(p.x),cy:y(p.y),r:5,class:'kl-dot',style:`--series:${s.color}`});
      });
    }
  }
  function gaussianInset(tooltip, kl) {
    const mu=Math.sqrt(2*kl);
    tooltip.classList.add('kl-gaussian-tooltip');
    el('b','',tooltip,String.raw`$\mathrm{KL}[p\,\|\,q]$ = ${num(kl)} nats`);
    const labels=el('div','kl-inset-labels',tooltip);
    el('span','',labels,String.raw`$p = \mathcal{N}(${mu.toFixed(2)}, 1)$`).style.color=colors.k1;
    el('span','',labels,String.raw`$q = \mathcal{N}(0, 1)$`).style.color=colors.k3;
    const width=244,height=128,m={l:14,r:8,t:9,b:23};
    const x=v=>m.l+(v+3.5)/9.5*(width-m.l-m.r);
    const y=v=>height-m.b-v/.44*(height-m.t-m.b);
    const svg=svgEl('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':`Gaussian densities p with mean ${mu.toFixed(2)} and q with mean 0, both with variance 1.`});
    tooltip.append(svg);
    svg.append(svgEl('line',{x1:m.l,x2:width-m.r,y1:y(0),y2:y(0),class:'kl-axis'}));
    [-3,0,3,6].forEach(t=>svg.append(svgEl('text',{x:x(t),y:height-6,'text-anchor':'middle',class:'kl-tick'},String(t))));
    [{mean:mu,color:colors.k1},{mean:0,color:colors.k3}].forEach(d=>{
      const points=Array.from({length:191},(_,i)=>{
        const v=-3.5+i*.05;return [x(v),y(Math.exp(-.5*(v-d.mean)**2)/Math.sqrt(2*Math.PI))];
      });
      const path=points.map(([px,py],i)=>`${i?'L':'M'}${px},${py}`).join(' ');
      svg.append(svgEl('path',{d:`M${x(-3.5)},${y(0)} L${path.slice(1)} L${x(6)},${y(0)} Z`,fill:d.color,opacity:'.12'}));
      svg.append(svgEl('path',{d:path,fill:'none',stroke:d.color,'stroke-width':2}));
    });
    const values=el('div','kl-inset-values',tooltip);
    el('span','',values,String.raw`$\operatorname{Var}_p(k_1)$ = ${num(2*kl)}`).style.color=colors.k1;
    el('span','',values,String.raw`$\operatorname{Var}_p(k_3)$ = ${num(Math.expm1(2*kl)-2*kl)}`).style.color=colors.k3;
  }
  function gaussian(root) {
    const leg=el('div','kl-legend',root),plot=el('div','kl-plot',root);
    const series=[{label:'k₁',color:colors.k1,data:[]},{label:'k₃',color:colors.k3,data:[]}];
    for(let i=2;i<=500;i++) {const mu=i*.005,d=mu*mu/2;series[0].data.push({x:d,y:mu*mu});series[1].data.push({x:d,y:Math.expm1(mu*mu)-mu*mu});}
    const draw=()=>{
      legend(leg,series);
      lineChart(plot,{series,xDomain:[0,3.125],xTicks:[0,1,2,3],logY:true,xLabel:'Exact KL[p ∥ q] (nats)',yLabel:'Estimator variance (nats²)',description:'Equal-variance Gaussians: hover to see p and q at each KL[p || q].',dots:false,renderTooltip:gaussianInset});
    };
    draw();return draw;
  }
  function categorical(root) {
    const q=[.1,.45,.45], cases=[{p:[.001,.4995,.4995],name:'p₁ · probability hole',color:colors.LLM},{p:[.1,.6577534291783038,.2422465708216962],name:'p₂ · smooth shift',color:colors.PPO}];
    const grid=el('div','kl-categorical',root);
    cases.forEach((c,caseIndex)=>{
      const z=c.p.map((p,i)=>Math.log(p/q[i])),d=sum(c.p.map((p,i)=>p*z[i]));
      const v1=sum(c.p.map((p,i)=>p*(z[i]-d)**2));
      const v3=sum(c.p.map((p,i)=>p*(z[i]+q[i]/p-1-d)**2));
      const card=el('div','kl-distribution',grid);
      el('h4','',card,c.name);
      const klLabel=el('p','',card);
      const formula=el('span','kl-math kl-math-inline',klLabel);
      const latex=String.raw`\mathrm{KL}[p_${caseIndex+1}\,\|\,q]`;
      if(window.katex) window.katex.render(latex,formula,{displayMode:false,throwOnError:false});
      else formula.textContent=`KL[${caseIndex===0?'p₁':'p₂'} ∥ q]`;
      klLabel.append(document.createTextNode(` = ${d.toFixed(8)} nats`));
      c.p.forEach((p,i)=>{
        const row=el('div','kl-prob-row',card);el('span','',row,['A','B','C'][i]);
        const track=el('div','kl-prob-track',row);const fill=el('div','kl-prob-fill',track);
        fill.style.width=`${p*100}%`;fill.style.setProperty('--series',c.color);
        const ref=el('div','kl-prob-ref',track);ref.style.left=`${q[i]*100}%`;ref.title=`q = ${q[i]}`;
        el('span','kl-prob-value',row,p.toFixed(4));
      });
      const values=el('div','kl-readouts',card);
      readouts(values,[[String.raw`$\operatorname{Var}_{p_${caseIndex+1}}(k_1)$ (nats²)`,v1.toFixed(5),colors.k1],[String.raw`$\operatorname{Var}_{p_${caseIndex+1}}(k_3)$ (nats²)`,v3.toFixed(5),colors.k3]]);
      el('div','kl-winner',card,v3>v1 ? `k₁ has ${(v3/v1).toFixed(0)}× lower variance` : `k₃ has ${(v1/v3).toFixed(1)}× lower variance`);
    });
    el('p','kl-note',root,'Bar length is p; dotted marks show q = (0.10, 0.45, 0.45). Each bar uses a 0–1 probability scale.');
    return ()=>{};
  }
  // Exact three-outcome sums; p2 is matched to p1 by monotone bisection.
  function holePair(h,referenceMass=.1) {
    const b=(1-referenceMass)/2,q=[referenceMass,b,b],a=referenceMass*10**(-h),p1=[a,(1-a)/2,(1-a)/2];
    const kl=p=>sum(p.map((v,i)=>v*Math.log(v/q[i])));
    const target=kl(p1);
    let lo=0,hi=b-1e-12;
    for(let i=0;i<70;i++) {
      const d=(lo+hi)/2;
      if(kl([referenceMass,b+d,b-d])<target)lo=d;else hi=d;
    }
    const d=h===0?0:(lo+hi)/2,p2=[referenceMass,b+d,b-d];
    const variance=p=>{
      const z=p.map((v,i)=>Math.log(v/q[i])),mean=kl(p);
      return [sum(p.map((v,i)=>v*(z[i]-mean)**2)),sum(p.map((v,i)=>v*(z[i]+Math.expm1(-z[i])-mean)**2))];
    };
    return {h,q,p1,p2,kl1:target,kl2:kl(p2),v1:variance(p1),v2:variance(p2)};
  }
  function holeness(root,referenceMass=.2) {
    const leg=el('div','kl-legend',root),plot=el('div','kl-plot',root);
    const points=Array.from({length:350},(_,i)=>holePair((i+1)/100,referenceMass));
    const series=[
      {label:'p₁ · k₁',color:colors.k1,arm:'v1',metric:0},
      {label:'p₁ · k₃',color:colors.k3,arm:'v1',metric:1},
      {label:'p₂ · k₁',color:colors.k1,arm:'v2',metric:0,dashed:true},
      {label:'p₂ · k₃',color:colors.k3,arm:'v2',metric:1,dashed:true}
    ].map(s=>({...s,data:points.map(p=>({x:p.h,y:p[s.arm][s.metric]}))}));
    legend(leg,series);
    function inset(tooltip,value) {
      const p=holePair(value,referenceMass);
      tooltip.classList.add('kl-hole-tooltip');
      el('b','',tooltip,`Hole depth $h$ = ${value.toFixed(2)}`);
      el('div','',tooltip,String.raw`$\mathrm{KL}[p_i\,\|\,q]$ = ${p.kl1.toFixed(8)} nats`);
      const svg=svgEl('svg',{viewBox:'0 0 240 116',role:'img','aria-label':'Three groups of outcome probabilities: p1, p2, and q. Outcomes appear in the same order within each group.'});
      tooltip.append(svg);
      const cs=[colors.LLM,colors.PPO,'var(--kl-muted)'];
      [0,.5,1].forEach(v=>{
        svg.append(svgEl('line',{x1:24,x2:236,y1:94-v*78,y2:94-v*78,class:'kl-grid'}));
        svg.append(svgEl('text',{x:20,y:98-v*78,'text-anchor':'end',class:'kl-tick'},String(v)));
      });
      [p.p1,p.p2,p.q].forEach((dist,j)=>{
        dist.forEach((v,i)=>{
          svg.append(svgEl('rect',{x:34+j*68+i*13,y:94-v*78,width:10,height:v*78,fill:cs[j]}));
        });
        const groupLabel=svgEl('foreignObject',{x:32+j*68,y:97,width:40,height:19});
        const label=el('div','kl-hole-group-label',null,['$p_1$','$p_2$','$q$'][j]);
        label.style.color=cs[j];groupLabel.append(label);svg.append(groupLabel);
      });
      el('div','kl-hole-variance-heading',tooltip,'Variances (nats²)');
      const values=el('div','kl-hole-variances',tooltip);
      [p.v1,p.v2].forEach((variances,j)=>{
        [1,3].forEach((estimator,i)=>{
          el('span','',values,String.raw`$\operatorname{Var}_{p_${j+1}}(k_${estimator})$ = ${num(variances[i])}`).style.color=i===0?colors.k1:colors.k3;
        });
      });
    }
    function draw() {
      lineChart(plot,{series,xDomain:[0,3.5],xTicks:[0,1,2,3,3.5],logY:true,xLabel:'Hole depth h',yLabel:'Estimator variance (nats²)',description:'Exact variances for probability-hole and smooth-shift distributions matched in KL at each hole depth. Reference q=(0.2,0.4,0.4). Hover to compare p1, p2 and q.',dots:false,renderTooltip:inset});
    }
    draw();return draw;
  }
  // Exact on-policy gradients. The categorical parameter here is a probability;
  // its gradients and optimization updates are with respect to the logit.
  function gradientMetrics(family, value) {
    if (family === 'gaussian') return {kl:value*value/2, gradients:[value,0,value]};
    const a=value, g=a*(1-a)*Math.log(a/(1-a));
    return {kl:a*Math.log(2*a)+(1-a)*Math.log(2*(1-a)), gradients:[g,g+.5-a,a-.5]};
  }
  function gradientTrajectory(family, rate, steps) {
    const gaussian=family==='gaussian';
    const parameters=Array(3).fill(gaussian ? 1 : Math.log(4));
    return Array.from({length:steps+1},(_,step)=>{
      const states=parameters.map(t=>gaussian ? t : 1/(1+Math.exp(-t)));
      const metrics=states.map(v=>gradientMetrics(family,v));
      if(step<steps) parameters.forEach((t,i)=>{parameters[i]=t-rate*metrics[i].gradients[i];});
      return {step,states,metrics};
    });
  }
  function gradientDistributions(tooltip, family, policies) {
    const gaussian=family==='gaussian', width=248, height=86;
    const svg=svgEl('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':gaussian ? 'Policy and reference Gaussian densities. The reference is gray.' : 'Action A and B probabilities for each policy. The reference is gray.'});
    tooltip.append(svg);
    const distributions=[{value:gaussian?0:.5,color:'var(--kl-muted)',dashed:true},...policies];
    if(gaussian) {
      const x=v=>12+(v+5)/10*224, y=v=>64-v/.44*52;
      svg.append(svgEl('line',{x1:12,x2:236,y1:64,y2:64,class:'kl-axis'}));
      [-4,0,4].forEach(v=>svg.append(svgEl('text',{x:x(v),y:81,'text-anchor':'middle',class:'kl-tick'},String(v))));
      distributions.forEach(p=>{
        const points=Array.from({length:201},(_,i)=>{const v=-5+i*.05;return [x(v),y(Math.exp(-.5*(v-p.value)**2)/Math.sqrt(2*Math.PI))];});
        const path=points.map(([px,py],i)=>`${i?'L':'M'}${px},${py}`).join(' ');
        if(policies.length===1 && !p.dashed) svg.append(svgEl('path',{d:`M12,64 L${path.slice(1)} L236,64 Z`,fill:p.color,opacity:'.1'}));
        svg.append(svgEl('path',{d:path,fill:'none',stroke:p.color,'stroke-width':2,...(p.dashed?{'stroke-dasharray':'4 3'}:{})}));
      });
    } else {
      [0,.5,1].forEach(v=>{
        svg.append(svgEl('line',{x1:25,x2:238,y1:64-v*52,y2:64-v*52,class:'kl-grid'}));
        svg.append(svgEl('text',{x:20,y:68-v*52,'text-anchor':'end',class:'kl-tick'},String(v)));
      });
      // Reference marks remain visible when policies equal q.
      [0,1].forEach(action=>{
        const center=80+action*110, bw=policies.length===1?23:16;
        policies.forEach((p,i)=>{
          const value=action===0?p.value:1-p.value;
          const x=center+(i-(policies.length-1)/2)*(bw+4)-bw/2;
          svg.append(svgEl('rect',{x,y:64-value*52,width:bw,height:value*52,fill:p.color,opacity:p.dashed?'.55':'.85'}));
        });
        svg.append(svgEl('line',{x1:center-37,x2:center+37,y1:38,y2:38,stroke:'var(--kl-muted)','stroke-width':1.8,'stroke-dasharray':'4 3'}));
        svg.append(svgEl('text',{x:center,y:81,'text-anchor':'middle',class:'kl-tick'},action===0?'A':'B'));
      });
    }
    svg.append(svgEl('text',{x:236,y:9,'text-anchor':'end',class:'kl-tick'},'q · dashed'));
  }
  function policyGradients(root, family) {
    const gaussian=family==='gaussian', rate=gaussian?.15:.5, steps=gaussian?30:80;
    const methods=[
      {label:'k₁ · reward',color:colors.k1},
      {label:'k₃ · reward',color:colors.k3},
      {label:'k₃ · loss',color:'var(--kl-green)',dashed:true}
    ];
    const leg=el('div','kl-legend',root),grid=el('div','kl-gradient-grid',root);
    const gradientPlot=el('div','kl-plot',grid),trainingPlot=el('div','kl-plot',grid);
    legend(leg,methods);
    const values=Array.from({length:301},(_,i)=>gaussian ? -2+4*i/300 : .01+.98*i/300);
    const gradientSeries=methods.map((s,i)=>({...s,data:values.map(value=>({x:value,y:gradientMetrics(family,value).gradients[i]}))}));
    const trajectory=gradientTrajectory(family,rate,steps);
    const trainingSeries=methods.map((s,i)=>({...s,data:trajectory.map(p=>({x:p.step,y:p.metrics[i].kl}))}));
    function gradientInset(tooltip,value) {
      const metrics=gradientMetrics(family,value);
      tooltip.classList.add('kl-gradient-tooltip');
      el('b','',tooltip,`${gaussian?'$\\mu$':'$a$'} = ${num(value)}`);
      gradientDistributions(tooltip,family,[{value,color:colors.k1}]);
      el('div','kl-gradient-equality',tooltip,String.raw`$\mathbb{E}_p[k_1]=\mathbb{E}_p[k_3]$ = ${num(metrics.kl)}`);
      const rows=el('div','kl-gradient-values',tooltip);
      methods.forEach((s,i)=>{
        const row=el('div','',rows);row.style.color=s.color;
        el('span','',row,s.label);el('span','',row,num(metrics.gradients[i]));
      });
    }
    function trainingInset(tooltip,step) {
      const point=trajectory[step];
      tooltip.classList.add('kl-gradient-tooltip');
      el('b','',tooltip,`Step ${step}`);
      gradientDistributions(tooltip,family,methods.map((s,i)=>({...s,value:point.states[i]})));
      const rows=el('div','kl-gradient-values kl-gradient-training-values',tooltip);
      const heading=el('div','kl-gradient-inset-note',rows);
      el('span','',heading,'Policy');el('span','',heading,gaussian?String.raw`$\mu$`:'$a$');el('span','',heading,'KL[p ∥ q]');
      methods.forEach((s,i)=>{
        const row=el('div','',rows);row.style.color=s.color;
        el('span','',row,s.label);el('span','',row,num(point.states[i]));el('span','',row,num(point.metrics[i].kl));
      });
    }
    function draw() {
      lineChart(gradientPlot,{series:gradientSeries,xDomain:gaussian?[-2,2]:[0,1],xTicks:gaussian?[-2,-1,0,1,2]:[0,.25,.5,.75,1],yDomain:gaussian?[-2.2,2.2]:[-.55,.55],yTicks:gaussian?[-2,-1,0,1,2]:[-.5,-.25,0,.25,.5],xLabel:gaussian?'Policy mean μ':'Action probability a',yLabel:'Expected penalty gradient',description:`${gaussian?'Gaussian':'Two-action'} policy: exact expected gradients for k1 in reward, k3 in reward, and k3 as direct loss.`,dots:false,floatingTooltip:true,renderTooltip:gradientInset});
      lineChart(trainingPlot,{series:trainingSeries,xDomain:[0,steps],xTicks:gaussian?[0,10,20,30]:[0,20,40,60,80],yDomain:gaussian?[0,.55]:[0,.7],yTicks:gaussian?[0,.1,.2,.3,.4,.5]:[0,.2,.4,.6],xLabel:'Gradient steps',yLabel:'True KL[p ∥ q] (nats)',description:`${gaussian?'Gaussian':'Two-action'} policy: true KL during exact gradient updates with learning rate ${rate}.`,dots:false,floatingTooltip:true,renderTooltip:trainingInset});
    }
    draw();return draw;
  }
  function gaussianGradients(root) {return policyGradients(root,'gaussian');}
  function categoricalGradients(root) {return policyGradients(root,'categorical');}
  function toyFigure(root, data, family) {
    // Three panels: KL-estimate variance and per-sample gradient variance
    // along the policy family, then true KL during minibatch gradient descent.
    const d=data[family], gaussian=family==='gaussian';
    const methods=[
      {key:'k1_reward',label:'k₁ · reward',color:colors.k1},
      {key:'k3_reward',label:'k₃ · reward',color:colors.k3},
      {key:'k3_loss',label:'k₃ · loss',color:'var(--kl-green)',dashed:true}
    ];
    const leg=el('div','kl-legend',root),grid=el('div','kl-gradient-grid kl-toy-grid',root);
    const valuePlot=el('div','kl-plot',grid),gradientPlot=el('div','kl-plot',grid),trainingPlot=el('div','kl-plot kl-toy-wide',grid);
    legend(leg,methods);
    const valueDomain=gaussian?[1e-6,1e2]:[1e-6,1e2], gradientDomain=gaussian?[1e-3,1e3]:[1e-6,1e2];
    const along=(key,domain)=>d.variances.map(r=>({x:r.x,y:r[key]>=domain[0]?r[key]:null,kl:r.kl}));
    const valueSeries=[
      {label:'k₁',color:colors.k1,data:along('k1',valueDomain)},
      {label:'k₃',color:colors.k3,data:along('k3',valueDomain)}
    ];
    const gradientSeries=methods.map(m=>({...m,data:along(m.key,gradientDomain)}));
    const trainingSeries=methods.map(m=>({...m,data:d.training[m.key].median.map((y,i)=>({x:i,y,lo:d.training[m.key].q25[i],hi:d.training[m.key].q75[i]}))}));
    const xName=gaussian?'μ':'a', xLabel=gaussian?'Policy mean μ':'Probability of the first action a';
    const variancesTip=series=>(tooltip,x)=>{
      tooltip.classList.add('kl-gradient-tooltip');
      const row=d.variances.find(r=>r.x===x);
      el('b','',tooltip,`${gaussian?'$\\mu$':'$a$'} = ${num(x)} · $\\mathrm{KL}[p\\,\\|\\,q]$ = ${num(row.kl)}`);
      const rows=el('div','kl-gradient-values',tooltip);
      series.forEach(s=>{
        const p=s.data.find(p=>p.x===x),r=el('div','',rows);r.style.color=s.color;
        el('span','',r,s.label);el('span','',r,p&&p.y!=null?num(p.y):'≈ 0');
      });
    };
    const trainingTip=(tooltip,step)=>{
      tooltip.classList.add('kl-gradient-tooltip');
      el('b','',tooltip,`Step ${step} · median KL[p ∥ q]`);
      const rows=el('div','kl-gradient-values',tooltip);
      trainingSeries.forEach(s=>{
        const p=s.data[step],r=el('div','',rows);r.style.color=s.color;
        el('span','',r,s.label);el('span','',r,num(p.y));
      });
      el('div','kl-gradient-equality',tooltip,'Middle 50% of runs shaded');
    };
    const xd=gaussian?[0,2]:[0,1], xt=gaussian?[0,.5,1,1.5,2]:[0,.25,.5,.75,1];
    const logTicks=dm=>{const t=[];for(let e=Math.log10(dm[0]);e<=Math.log10(dm[1])+1e-9;e+=2)t.push(10**e);return t;};
    function draw() {
      lineChart(valuePlot,{series:valueSeries,xDomain:xd,xTicks:xt,logY:true,yDomain:valueDomain,yTicks:logTicks(valueDomain),xLabel,yLabel:'Variance of KL estimate',description:`${gaussian?'Gaussian':'Two-action'} policy: exact variance of the k1 and k3 KL estimates. The marker shows where training starts.`,dots:false,selectedX:d.start,floatingTooltip:true,renderTooltip:variancesTip(valueSeries)});
      lineChart(gradientPlot,{series:gradientSeries,xDomain:xd,xTicks:xt,logY:true,yDomain:gradientDomain,yTicks:logTicks(gradientDomain),xLabel,yLabel:'Variance of gradient',description:`${gaussian?'Gaussian':'Two-action'} policy: exact per-sample gradient variance for k1 in reward, k3 in reward, and k3 as direct loss.`,dots:false,selectedX:d.start,floatingTooltip:true,renderTooltip:variancesTip(gradientSeries)});
      lineChart(trainingPlot,{series:trainingSeries,xDomain:[0,d.steps],xTicks:[0,10,20,30,40,50,60],yDomain:[0,gaussian?1.4:.75],yTicks:gaussian?[0,.25,.5,.75,1,1.25]:[0,.2,.4,.6],xLabel:'Gradient steps',yLabel:'True KL[p ∥ q] (nats)',description:`${gaussian?'Gaussian':'Two-action'} policy: median true KL over ${d.seeds} runs of minibatch gradient descent, batch ${d.batch}, learning rate ${d.rate}. Shading spans the middle 50% of runs.`,dots:false,floatingTooltip:true,renderTooltip:trainingTip});
    }
    draw();return draw;
  }
  function gaussianToy(root,data) {return toyFigure(root,data,'gaussian');}
  function categoricalToy(root,data) {return toyFigure(root,data,'categorical');}
  function empirical(root,data) {
    const leg=el('div','kl-legend',root),plot=el('div','kl-plot',root);
    const draw=()=>{
      const rows=data.bins.bins.filter(r=>r.view==='full_range');
      const series=['LLM','PPO'].flatMap(domain=>['k1','k3'].map(metric=>({
        label:`${domain} · ${metric==='k1'?'k₁':'k₃'}`,color:colors[domain],dashed:metric==='k3',
        data:rows.filter(r=>r.domain===domain).map(r=>({x:r.x,y:r.displayed?r[`median_${metric}`]:null,lo:r[`q25_${metric}`],hi:r[`q75_${metric}`],count:r.count,bin:[r.lower_kl,r.upper_kl]}))
      })));
      const edges=data.bins.full_edges;
      legend(leg,series);
      lineChart(plot,{series,xDomain:[edges[0],edges.at(-1)],logX:true,logY:true,xLabel:'Exact KL[p ∥ q] (nats)',yLabel:'Estimator variance (nats²)',description:'Full-range raw k1 and k3 conditional variances for LLM and PPO policy pairs. Shading shows within-bin interquartile ranges.'});
    };
    draw();return draw;
  }
  function rewards(root,data) {
    let env='LLM';
    const controls=el('div','kl-controls',root);
    tabs(controls,[['LLM','LLM · AIME24'],['Hopper','Hopper'],['HalfCheetah','HalfCheetah'],['Walker2d','Walker2d']],'LLM',v=>{env=v;draw();});
    const leg=el('div','kl-legend',root),plot=el('div','kl-plot',root),note=el('p','kl-note',root);
    const draw=()=>{
      const d=data.rewards[env],llm=env==='LLM';
      const series=['k1','k3'].map(arm=>({label:arm==='k1'?'k₁':'k₃',color:colors[arm],data:d.x.map((x,i)=>{
        const a=d.arms[arm].map(seed=>seed[i]);return{x,y:mean(a),lo:quantile(a,.25),hi:quantile(a,.75)};
      })}));
      const reference=llm?{y:d.teacher,label:'Teacher'}:null;
      legend(leg,series,reference);
      lineChart(plot,{series,reference,xTicks:llm?d.x:undefined,yDomain:llm?[0,1]:undefined,percent:llm,xLabel:llm?'Completed training updates':'Environment interactions',yLabel:llm?'AIME2024 pass@1':'Held-out episode return',description:`${env} reward curves: means over ${d.seeds} training seeds, with interquartile spread across seed means.`});
      note.textContent=llm?'Completed four-update study · 3 paired seeds · 120 evaluation responses per checkpoint. Longer runs are in progress.':'5 seeds per method · 20 held-out episodes per seed and checkpoint · equal environment interactions, not equal optimizer steps.';
    };
    draw();return draw;
  }
  async function init() {
    document.querySelectorAll('.kl-math').forEach(node=>{
      if (window.katex) window.katex.render(node.textContent.trim(),node,{displayMode:node.dataset.display!=='false',throwOnError:false,strict:'warn'});
    });
    document.querySelectorAll('.kl-code-body').forEach(body=>{
      const button=el('button','kl-copy',body,'Copy code');button.type='button';
      button.addEventListener('click',async()=>{
        try {await navigator.clipboard.writeText(body.querySelector('code').textContent);button.textContent='Copied';}
        catch {button.textContent='Select code to copy';}
        setTimeout(()=>{button.textContent='Copy code';},1800);
      });
    });
    const dataPromises={};
    for(const host of document.querySelectorAll('.kl-viz')) {
      const root=host.querySelector('.kl-interactive'),kind=host.dataset.chart;
      try {
        let data;
        if (['empirical','rewards','gaussianToy','categoricalToy'].includes(kind)) {
          const source=host.dataset.source;
          dataPromises[source] ||= fetch(source).then(r=>{if(!r.ok)throw Error(`Data fetch: ${r.status}`);return r.json();});
          data=await dataPromises[source];
        }
        root.replaceChildren();
        const redraw=({gaussian,categorical,holeness,gaussianGradients,categoricalGradients,gaussianToy,categoricalToy,empirical,rewards})[kind](root,data);
        let oldWidth=host.clientWidth,frame;
        new ResizeObserver(()=>{
          if(host.clientWidth!==oldWidth) {oldWidth=host.clientWidth;cancelAnimationFrame(frame);frame=requestAnimationFrame(redraw);}
        }).observe(host);
      } catch(error) {
        root.replaceChildren();el('p','kl-note',root,'The interactive figure could not load. Numerical results and reproducible code are below.');console.error(error);
      }
    }
  }
  init();
})();
