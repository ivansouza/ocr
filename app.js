
// ===== CONSTANTES DO MODELO =====
// Média e std do pré-processamento PaddleOCR
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// Character dictionary do PP-OCRv6 (extraído do inference.yml)
let CHARS = [];

// Carregar dicionário
fetch('chars.json').then(r=>r.json()).then(d=>{CHARS=d;console.log('Dict:',CHARS.length)}).catch(e=>console.error('Dict:',e));

let detSession = null;
let recSession = null;
let modelsLoaded = false;
let currentFile = null;

// ===== CARREGAR MODELOS =====
function updateProgress(pct, step, size) {
  document.getElementById('progressContainer').classList.add('show');
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressPct').textContent = pct + '%';
  document.getElementById('progressSize').textContent = size;
  document.getElementById('progressStep').textContent = step;
}

async function loadModels() {
  const statusEl = document.getElementById('modelStatus');
  try {
    statusEl.className = 'model-status loading';
    statusEl.innerHTML = '<div class="spinner"></div><span>Preparando download dos modelos...</span>';
    updateProgress(0, 'Iniciando...', '0 / 30.5 MB');
    
    // Baixar modelo de detecção (9.5MB)
    updateProgress(5, 'Baixando modelo de detecção (9.5 MB)...', '0 / 30.5 MB');
    statusEl.innerHTML = '<div class="spinner"></div><span>Baixando detecção (9.5MB)...</span>';
    
    console.log('Iniciando download: small_det.onnx');
    const detResponse = await fetch('small_det.onnx');
    if (!detResponse.ok) throw new Error('Falha ao baixar detecção: HTTP ' + detResponse.status);
    console.log('Download detecção concluído, tamanho:', detResponse.headers.get('content-length'));
    const detBlob = await detResponse.blob();
    console.log('Blob detecção:', detBlob.size, 'bytes');
    updateProgress(35, 'Modelo de detecção baixado. Carregando no ONNX Runtime...', '9.5 / 30.5 MB');
    statusEl.innerHTML = '<div class="spinner"></div><span>Carregando detecção no ONNX Runtime...</span>';
    
    const detArrayBuffer = await detBlob.arrayBuffer();
    detSession = await ort.InferenceSession.create(detArrayBuffer, {
      executionProviders: ['webgl'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false
    });
    console.log('Detecção inputs:', Object.keys(detSession.inputNames), 'outputs:', Object.keys(detSession.outputNames));
    
    updateProgress(40, 'Detecção carregada. Baixando modelo de reconhecimento (21 MB)...', '9.5 / 30.5 MB');
    statusEl.innerHTML = '<div class="spinner"></div><span>Baixando reconhecimento (21MB)...</span>';
    
    console.log('Iniciando download: small_rec.onnx');
    const recResponse = await fetch('small_rec.onnx');
    if (!recResponse.ok) throw new Error('Falha ao baixar reconhecimento: HTTP ' + recResponse.status);
    console.log('Download reconhecimento concluído');
    const recBlob = await recResponse.blob();
    console.log('Blob reconhecimento:', recBlob.size, 'bytes');
    updateProgress(75, 'Modelo de reconhecimento baixado. Carregando no ONNX Runtime...', '30.5 / 30.5 MB');
    statusEl.innerHTML = '<div class="spinner"></div><span>Carregando reconhecimento no ONNX Runtime...</span>';
    
    const recArrayBuffer = await recBlob.arrayBuffer();
    recSession = await ort.InferenceSession.create(recArrayBuffer, {
      executionProviders: ['webgl'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false
    });
    console.log('Reconhecimento inputs:', Object.keys(recSession.inputNames), 'outputs:', Object.keys(recSession.outputNames));
    
    updateProgress(100, '✅ Modelos prontos!', '30.5 / 30.5 MB');
    modelsLoaded = true;
    statusEl.className = 'model-status ready';
    statusEl.innerHTML = '✅ Modelos ONNX carregados (det: 9.5MB + rec: 21MB)';
    document.getElementById('btnProcess').disabled = !currentFile;
    // Esconder o spinner antigo
    const oldSpinner = document.querySelector('.model-status.loading');
    if (oldSpinner) oldSpinner.className = 'model-status ready';
  } catch (err) {
    statusEl.className = 'model-status error';
    const msg = err.message || err.toString() || 'Erro desconhecido';
    statusEl.innerHTML = '❌ Erro: ' + msg;
    updateProgress(0, '❌ ' + msg, 'Falhou');
    console.error('Erro detalhado:', err);
    console.error('Stack:', err.stack);
    // Mostrar erro na tela
    const errDiv = document.createElement('div');
    errDiv.style.cssText = 'margin-top:0.5rem;padding:0.5rem;background:rgba(239,68,68,0.1);border-radius:6px;font-size:0.75rem;color:var(--red);word-break:break-all;font-family:monospace;';
    errDiv.textContent = err.stack || err.message || 'Erro desconhecido';
    statusEl.parentNode.appendChild(errDiv);
  }
}

// ===== PRÉ-PROCESSAMENTO =====
function preprocessImage(img) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  // Redimensionar mantendo proporção (altura máxima 736 para detecção)
  const maxSide = 736;
  let w = img.width, h = img.height;
  if (w > maxSide || h > maxSide) {
    const scale = maxSide / Math.max(w, h);
    w = Math.round(w * scale);
    h = Math.round(h * scale);
  }
  
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(img, 0, 0, w, h);
  
  const imageData = ctx.getImageData(0, 0, w, h);
  const pixels = imageData.data;
  
  // Converter para tensor (NCHW) com normalização
  const tensor = new Float32Array(1 * 3 * h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const r = pixels[idx] / 255;
      const g = pixels[idx + 1] / 255;
      const b = pixels[idx + 2] / 255;
      tensor[0 * h * w + y * w + x] = (r - MEAN[0]) / STD[0];
      tensor[1 * h * w + y * w + x] = (g - MEAN[1]) / STD[1];
      tensor[2 * h * w + y * w + x] = (b - MEAN[2]) / STD[2];
    }
  }
  
  return { tensor, width: w, height: h, canvas };
}

// ===== PÓS-PROCESSAMENTO DA DETECÇÃO (DBPostProcess simplificado) =====
function postProcessDet(output, origW, origH, imgW, imgH) {
  // output é um tensor [1, 1, H, W] com scores
  const data = output.data;
  const shape = output.dims;
  const h = shape[2], w = shape[3];
  
  // Threshold
  const thresh = 0.2;
  const boxThresh = 0.4;
  
  // Encontrar regiões de texto (simplificado - bounding boxes)
  const regions = [];
  const visited = new Uint8Array(h * w);
  
  // Flood fill simples pra encontrar componentes conectados
  function floodFill(sx, sy) {
    const stack = [[sx, sy]];
    const pixels = [];
    let minX = sx, maxX = sx, minY = sy, maxY = sy;
    
    while (stack.length) {
      const [cx, cy] = stack.pop();
      const idx = cy * w + cx;
      if (cx < 0 || cx >= w || cy < 0 || cy >= h || visited[idx] || data[idx] <= thresh) continue;
      visited[idx] = 1;
      pixels.push([cx, cy]);
      minX = Math.min(minX, cx);
      maxX = Math.max(maxX, cx);
      minY = Math.min(minY, cy);
      maxY = Math.max(maxY, cy);
      stack.push([cx-1,cy], [cx+1,cy], [cx,cy-1], [cx,cy+1]);
    }
    
    if (pixels.length < 3) return null;
    
    // Calcular score médio
    let score = 0;
    for (const [px, py] of pixels) score += data[py * w + px];
    score /= pixels.length;
    
    if (score < boxThresh) return null;
    
    // Escalar de volta pra imagem original
    const scaleX = origW / imgW;
    const scaleY = origH / imgH;
    
    return {
      box: [
        minX * scaleX, minY * scaleY,
        maxX * scaleX, minY * scaleY,
        maxX * scaleX, maxY * scaleY,
        minX * scaleX, maxY * scaleY
      ],
      score
    };
  }
  
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (!visited[idx] && data[idx] > thresh) {
        const region = floodFill(x, y);
        if (region) regions.push(region);
      }
    }
  }
  
  // Ordenar por posição (cima pra baixo)
  regions.sort((a, b) => a.box[1] - b.box[1]);
  
  return regions;
}

// ===== RECORTAR E RECONHECER =====
async function recognizeRegions(canvas, regions) {
  const ctx = canvas.getContext('2d');
  const results = [];
  
  for (const region of regions) {
    const [x1, y1, x2, y2, x3, y3, x4, y4] = region.box;
    const rx = Math.max(0, Math.min(x1, x2, x3, x4));
    const ry = Math.max(0, Math.min(y1, y2, y3, y4));
    const rw = Math.min(canvas.width, Math.max(x1, x2, x3, x4)) - rx;
    const rh = Math.min(canvas.height, Math.max(y1, y2, y3, y4)) - ry;
    
    if (rw < 2 || rh < 2) continue;
    
    // Recortar
    const cropCanvas = document.createElement('canvas');
    const cropCtx = cropCanvas.getContext('2d');
    
    // Redimensionar pra altura 48 (input do rec)
    const recH = 48;
    const recW = Math.max(32, Math.round(rw / rh * recH));
    cropCanvas.width = recW;
    cropCanvas.height = recH;
    cropCtx.drawImage(canvas, rx, ry, rw, rh, 0, 0, recW, recH);
    
    // Converter pra tensor
    const imgData = cropCtx.getImageData(0, 0, recW, recH);
    const pixels = imgData.data;
    const tensor = new Float32Array(1 * 3 * recH * recW);
    for (let y = 0; y < recH; y++) {
      for (let x = 0; x < recW; x++) {
        const idx = (y * recW + x) * 4;
        tensor[0 * recH * recW + y * recW + x] = pixels[idx] / 255;
        tensor[1 * recH * recW + y * recW + x] = pixels[idx + 1] / 255;
        tensor[2 * recH * recW + y * recW + x] = pixels[idx + 2] / 255;
      }
    }
    
    try {
      const recInNames = recSession.inputNames;
      const recInName = Array.isArray(recInNames) ? recInNames[0] : Object.values(recInNames)[0];
      const feeds = {};
      feeds[recInName] = new ort.Tensor('float32', tensor, [1, 3, recH, recW]);
      const outputs = await recSession.run(feeds);
      const recOutNames = Object.keys(outputs);
      const recOutName = Array.isArray(recSession.outputNames) ? recSession.outputNames[0] : Object.values(recSession.outputNames)[0];
      const outData = outputs[recOutName].data;
      const outShape = outputs[recOutName].dims;
      
      // Decodificar CTC (índice 0 = blank)
      let text = '';
      let prevChar = -1;
      let confSum = 0;
      let confCount = 0;
      
      for (let t = 0; t < outShape[1]; t++) {
        let maxVal = -Infinity;
        let maxIdx = 0;
        for (let c = 0; c < outShape[2]; c++) {
          const val = outData[t * outShape[2] + c];
          if (val > maxVal) { maxVal = val; maxIdx = c; }
        }
        // Softmax pra confiança
        let expSum = 0;
        for (let c = 0; c < outShape[2]; c++) {
          expSum += Math.exp(outData[t * outShape[2] + c] - maxVal);
        }
        const conf = Math.exp(maxVal) / expSum;
        
        // CTC blank = índice 0. Pular blanks e repetições
        // Limitar ao tamanho real do dicionário (ignorar padding)
        if (maxIdx > 0 && maxIdx !== prevChar) {
          const charIdx = maxIdx - 1;
          if (charIdx < CHARS.length) {
            const ch = CHARS[charIdx];
            // Manter apenas caracteres úteis para PT-BR (ignorar chinês, japonês, etc.)
            const code = ch.charCodeAt(0);
            const isLatin = (code >= 32 && code <= 126) || // ASCII imprimível
                           (code >= 192 && code <= 255) || // Latin-1 Supplement (acentos)
                           code === 10 || code === 13;    // newline
            if (isLatin) {
              text += ch;
              confSum += conf;
              confCount++;
            }
          }
        }
        prevChar = maxIdx;
      }
      
      const avgConf = confCount > 0 ? confSum / confCount : 0;
      
      if (text.trim()) {
        results.push({ text: text.trim(), confidence: avgConf });
      }
    } catch (err) {
      console.warn('Erro no reconhecimento:', err);
    }
  }
  
  return results;
}

// ===== EXTRAIR CAMPOS =====
function extractFields(texts) {
  const campos = {};
  const fullText = texts.join('\n');
  
  // CNPJ
  for (const t of texts) {
    const m = t.match(/\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}/);
    if (m) { campos['CNPJ'] = m[0]; break; }
  }
  
  // Valor
  for (const t of texts) {
    const m = t.match(/R?\$?\s*[\d.]+\,[\d]{2}/);
    if (m) { campos['VALOR'] = m[0].trim(); break; }
  }
  
  // Data
  for (const t of texts) {
    const m = t.match(/\d{2}\w{3}\s*\d{4}/);
    if (m) { campos['DATA'] = m[0]; break; }
  }
  
  // Chave Pix
  for (const t of texts) {
    const m = t.match(/\+55\d{10,11}/);
    if (m) { campos['CHAVE_PIX'] = m[0]; break; }
  }
  
  return campos;
}

// ===== UI =====
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewArea = document.getElementById('previewArea');
const preview = document.getElementById('preview');
const btnProcess = document.getElementById('btnProcess');
const statusBar = document.getElementById('statusBar');
const statusText = document.getElementById('statusText');
const resultCard = document.getElementById('resultCard');
const fieldsContainer = document.getElementById('fieldsContainer');
const textLines = document.getElementById('textLines');

fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    previewArea.style.display = 'block';
    btnProcess.disabled = !modelsLoaded;
  };
  reader.readAsDataURL(file);
});

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event('change'));
  }
});

btnProcess.addEventListener('click', processImage);

async function processImage() {
  if (!currentFile || !modelsLoaded) return;
  
  statusBar.className = 'status-bar loading';
  statusText.textContent = 'Pré-processando imagem...';
  btnProcess.disabled = true;
  
  try {
    // Carregar imagem
    const img = await new Promise((resolve, reject) => {
      const i = new Image();
      i.onload = () => resolve(i);
      i.onerror = reject;
      i.src = URL.createObjectURL(currentFile);
    });
    
    statusText.textContent = 'Detectando regiões de texto...';
    const { tensor, width, height, canvas } = preprocessImage(img);
    
    const detInNames = detSession.inputNames;
    console.log('Det inputNames type:', typeof detInNames, 'value:', JSON.stringify(detInNames));
    const detInName = Array.isArray(detInNames) ? detInNames[0] : Object.values(detInNames)[0];
    const feeds = {};
    feeds[detInName] = new ort.Tensor('float32', tensor, [1, 3, height, width]);
    updateProgress(30, 'Rodando detecção de texto no modelo ONNX...', '');
    statusText.textContent = 'Rodando detecção...';
    let detOutputs;
    try {
      detOutputs = await detSession.run(feeds);
    } catch (runErr) {
      console.error('detSession.run ERROR:', runErr);
      console.error('detSession.run name:', runErr.name);
      console.error('detSession.run message:', runErr.message);
      console.error('detSession.run code:', runErr.code);
      console.error('Tensor shape:', [1, 3, height, width], 'size:', tensor.length);
      console.error('Input name:', detInName);
      throw runErr;
    }
    // Pega o nome do primeiro output dinamicamente
    const detOutName = Array.isArray(detSession.outputNames) ? detSession.outputNames[0] : Object.values(detSession.outputNames)[0];
    console.log('Det input:', detInName, 'output:', detOutName, 'shape:', detOutputs[detOutName].dims);
    
    updateProgress(50, 'Processando regiões detectadas...', '');
    statusText.textContent = 'Processando regiões detectadas...';
    const regions = postProcessDet(detOutputs[detOutName], img.width, img.height, width, height);
    
    if (regions.length === 0) {
      throw new Error('Nenhuma região de texto detectada.');
    }
    
    statusText.textContent = `Reconhecendo ${regions.length} regiões de texto...`;
    updateProgress(60, `Reconhecendo ${regions.length} regiões de texto...`, '');
    const results = await recognizeRegions(canvas, regions);
    
    if (results.length === 0) {
      throw new Error('Nenhum texto reconhecido.');
    }
    
    updateProgress(100, `✅ ${results.length} linhas extraídas!`, '');
    statusBar.className = 'status-bar done';
    statusText.textContent = `✅ ${results.length} linhas extraídas`;
    showResults(results);
  } catch (err) {
    statusBar.className = 'status-bar error';
    const errMsg = (err && err.message) ? err.message : (err ? err.toString() : 'Erro desconhecido');
    statusText.textContent = '❌ ' + errMsg;
    console.error(err);
  } finally {
    btnProcess.disabled = false;
  }
}

function showResults(results) {
  resultCard.classList.add('show');
  
  const texts = results.map(r => r.text);
  const campos = extractFields(texts);
  
  let html = '';
  for (const [key, val] of Object.entries(campos)) {
    html += `<div class="field"><span class="label">${key}</span><span class="value">${val}</span></div>`;
  }
  fieldsContainer.innerHTML = html || '<p style="color:var(--muted)">Nenhum campo estruturado identificado.</p>';
  
  let linesHtml = '';
  results.forEach(r => {
    const pct = Math.round(r.confidence * 100);
    const cls = pct >= 90 ? 'high' : pct >= 70 ? 'mid' : 'low';
    linesHtml += `<div class="text-line">
      <span>${r.text}</span>
      <div class="conf-bar"><div class="fill ${cls}" style="width:${pct}%"></div></div>
      <span style="font-size:0.7rem;color:var(--muted);width:28px;text-align:right">${pct}%</span>
    </div>`;
  });
  textLines.innerHTML = linesHtml;
}

// ===== INICIAR =====
loadModels();
