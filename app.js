let modelsLoaded = false;
let currentFile = null;

// Filtro: só ASCII imprimível (32-126) + Latin-1 Supplement acentos (192-255)
function filterChar(c) {
  const code = c.charCodeAt(0);
  return (code >= 32 && code <= 126) || (code >= 192 && code <= 255) || code === 10 || code === 13;
}

function cleanText(text) {
  return text.split('').filter(filterChar).join('').trim();
}

function updateProgress(pct, step, size) {
  const container = document.getElementById('progressContainer');
  const fill = document.getElementById('progressFill');
  const pctEl = document.getElementById('progressPct');
  const sizeEl = document.getElementById('progressSize');
  const stepEl = document.getElementById('progressStep');
  if (!container || !fill) return;
  container.classList.add('show');
  fill.style.width = pct + '%';
  if (pctEl) pctEl.textContent = pct + '%';
  if (sizeEl && size) sizeEl.textContent = size;
  if (stepEl && step) stepEl.textContent = step;
}

async function loadModels() {
  const statusEl = document.getElementById('modelStatus');
  try {
    statusEl.className = 'model-status loading';
    statusEl.innerHTML = '<div class="spinner"></div><span>Inicializando Tesseract.js...</span>';
    updateProgress(10, 'Inicializando Tesseract.js...', '');

    if (typeof Tesseract === 'undefined') {
      throw new Error('Tesseract.js não carregou. Verifique sua conexão.');
    }

    updateProgress(100, '✅ Tesseract.js pronto!', '');
    modelsLoaded = true;
    statusEl.className = 'model-status ready';
    statusEl.innerHTML = '✅ Tesseract.js pronto (OCR português)';
    document.getElementById('btnProcess').disabled = !currentFile;
  } catch (err) {
    statusEl.className = 'model-status error';
    const msg = (err && err.message) ? err.message : (err ? err.toString() : 'Erro desconhecido');
    statusEl.innerHTML = '❌ Erro: ' + msg;
    console.error(err);
  }
}

function extractFields(texts) {
  const campos = {};
  const fullText = texts.join('\n');

  for (const t of texts) {
    const m = t.match(/\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}/);
    if (m) { campos['CNPJ'] = m[0]; break; }
  }
  for (const t of texts) {
    const m = t.match(/R?\$?\s*[\d.]+\,[\d]{2}/);
    if (m) { campos['VALOR'] = m[0].trim(); break; }
  }
  for (const t of texts) {
    const m = t.match(/\d{2}\s*\w{3}\s*\d{4}/);
    if (m) { campos['DATA'] = m[0]; break; }
  }
  for (const t of texts) {
    const m = t.match(/\+55\d{10,11}/);
    if (m) { campos['CHAVE_PIX'] = m[0]; break; }
  }

  return campos;
}

async function processImage() {
  if (!currentFile || !modelsLoaded) return;

  const statusBar = document.getElementById('statusBar');
  const statusText = document.getElementById('statusText');
  const btnProcess = document.getElementById('btnProcess');

  statusBar.className = 'status-bar loading';
  statusText.textContent = 'Iniciando OCR...';
  btnProcess.disabled = true;
  updateProgress(5, 'Preparando imagem...', '');

  try {
    const img = await new Promise((resolve, reject) => {
      const i = new Image();
      i.onload = () => resolve(i);
      i.onerror = () => reject(new Error('Falha ao carregar imagem'));
      i.src = URL.createObjectURL(currentFile);
    });

    updateProgress(10, 'Enviando para Tesseract.js...', '');
    statusText.textContent = 'Reconhecendo texto...';

    const result = await Tesseract.recognize(
      img,
      'por',
      {
        logger: m => {
          if (m.status === 'recognizing text') {
            const pct = Math.round(m.progress * 100);
            updateProgress(10 + Math.round(pct * 0.8), 'Reconhecendo: ' + pct + '%', '');
            statusText.textContent = 'Reconhecendo texto... ' + pct + '%';
          }
        }
      }
    );

    updateProgress(95, 'Organizando resultados...', '');

    const lines = result.data.words
      .map(w => ({
        text: cleanText(w.text),
        confidence: w.confidence / 100,
        bbox: w.bbox
      }))
      .filter(w => w.text.length > 0);

    lines.sort((a, b) => a.bbox.y0 - b.bbox.y0);

    updateProgress(100, '✅ ' + lines.length + ' linhas extraídas!', '');
    statusBar.className = 'status-bar done';
    statusText.textContent = '✅ ' + lines.length + ' linhas extraídas';
    showResults(lines);
  } catch (err) {
    statusBar.className = 'status-bar error';
    const msg = (err && err.message) ? err.message : (err ? err.toString() : 'Erro desconhecido');
    console.error('ERRO:', err);
    statusText.textContent = '❌ ' + msg;
    updateProgress(0, '❌ ' + msg, '');
  } finally {
    btnProcess.disabled = false;
  }
}

function showResults(results) {
  const resultCard = document.getElementById('resultCard');
  const fieldsContainer = document.getElementById('fieldsContainer');
  const textLines = document.getElementById('textLines');
  resultCard.classList.add('show');

  const texts = results.map(r => r.text);
  const campos = extractFields(texts);

  let html = '';
  for (const [key, val] of Object.entries(campos)) {
    html += '<div class="field"><span class="label">' + key + '</span><span class="value">' + val + '</span></div>';
  }
  fieldsContainer.innerHTML = html || '<p style="color:var(--muted)">Nenhum campo estruturado identificado.</p>';

  let linesHtml = '';
  results.forEach(r => {
    const pct = Math.round(r.confidence * 100);
    const cls = pct >= 90 ? 'high' : pct >= 70 ? 'mid' : 'low';
    linesHtml += '<div class="text-line">' +
      '<span>' + r.text + '</span>' +
      '<div class="conf-bar"><div class="fill ' + cls + '" style="width:' + pct + '%"></div></div>' +
      '<span style="font-size:0.7rem;color:var(--muted);width:28px;text-align:right">' + pct + '%</span>' +
    '</div>';
  });
  textLines.innerHTML = linesHtml;
}

// ===== UI =====
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewArea = document.getElementById('previewArea');
const preview = document.getElementById('preview');
const btnProcess = document.getElementById('btnProcess');

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

// Iniciar
loadModels();
