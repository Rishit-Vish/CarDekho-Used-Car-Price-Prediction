/* ==========================================================
   Used car price estimator

   GET  /options  -> { Car_names, Brands, Model, Seller_type, fuel_type,
                       Transmission_type, Catalog? }
   POST /predict  -> { predicted_selling_price }

   Nothing about cars is hardcoded here: every list comes from /options.
   Dependent dropdowns need the optional Catalog field:
       "Catalog": { "<brand>": { "<model>": ["<car name>", ...] } }
   Without it, all three lists are shown unfiltered and a notice explains why.
   ========================================================== */
(() => {
  'use strict';

  /* ---------- Config ---------- */

  const DEFAULT_API = 'http://127.0.0.1:8000';
  const OPTIONS_TIMEOUT_MS = 15000;
  const PREDICT_TIMEOUT_MS = 30000;
  const MIN_LOADING_MS = 1100;   // lets the odometer spin long enough to read as "working"
  const MAX_RENDERED = 100;      // options drawn at once in a dropdown

  const COMBO_FIELDS = { brand: 'brand', model: 'model', car_name: 'car name' };
  const CHOICE_FIELDS = { seller_type: 'seller type', fuel_type: 'fuel type', transmission_type: 'transmission' };
  const NUMBER_FIELDS = {
    vehicle_age: { label: 'Vehicle age', ask: 'vehicle age', int: true, min: 0 },
    km_driven:   { label: 'Kilometres driven', ask: 'kilometres driven', int: true, min: 0 },
    mileage:     { label: 'Mileage', ask: 'mileage', int: false, min: 0 },
    engine:      { label: 'Engine capacity', ask: 'engine capacity', int: true, min: 0 },
    max_power:   { label: 'Maximum power', ask: 'maximum power', int: false, min: 0 },
    seats:       { label: 'Seats', ask: 'number of seats', int: true, min: 1 },
  };
  // Exactly the field names of the Pydantic Features schema
  const PAYLOAD_FIELDS = [
    'car_name', 'brand', 'model', 'vehicle_age', 'km_driven', 'seller_type',
    'fuel_type', 'transmission_type', 'mileage', 'engine', 'max_power', 'seats',
  ];
  const FOCUS_ORDER = [
    'brand', 'model', 'car_name', 'seller_type', 'fuel_type', 'transmission_type',
    'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
  ];
  const FIELD_NAMES = { brand: 'Brand', model: 'Model', car_name: 'Car name', seller_type: 'Seller type', fuel_type: 'Fuel type', transmission_type: 'Transmission' };
  Object.entries(NUMBER_FIELDS).forEach(([k, v]) => { FIELD_NAMES[k] = v.label; });

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- Helpers ---------- */

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const el = (tag, cls) => { const n = document.createElement(tag); if (cls) n.className = cls; return n; };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const cmp = (a, b) => a.localeCompare(b, 'en', { numeric: true, sensitivity: 'base' });
  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

  const nf = new Intl.NumberFormat('en-IN');
  const inr = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 });

  function priceInWords(v) {
    if (v >= 1e7) return `\u20B9${(v / 1e7).toFixed(2)} crore`;
    if (v >= 1e5) return `\u20B9${(v / 1e5).toFixed(2)} lakh`;
    return inr.format(v);
  }

  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
  };

  const form = $('#form');
  const gauge = $('#gauge');
  const submitBtn = $('#submitBtn');
  const submitText = $('#submitText');

  /* ---------- API address and status ---------- */

  const apiStatus = $('#apiStatus');
  const apiStatusText = $('#apiStatusText');
  const apiPanel = $('#apiPanel');
  const apiUrlInput = $('#apiUrl');

  let apiBase = (store.get('cardekho.api') || DEFAULT_API).replace(/\/+$/, '');
  apiUrlInput.value = apiBase;

  function setHealth(state) {
    apiStatus.dataset.state = state;
    apiStatusText.textContent = { online: 'API connected', offline: 'API offline', checking: 'Connecting' }[state];
  }

  function toggleApiPanel(open) {
    const next = open ?? apiPanel.hidden;
    apiPanel.hidden = !next;
    apiStatus.setAttribute('aria-expanded', String(next));
    if (next) apiUrlInput.focus();
  }
  apiStatus.addEventListener('click', () => toggleApiPanel());
  document.addEventListener('click', (e) => { if (!apiPanel.hidden && !e.target.closest('.api')) toggleApiPanel(false); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !apiPanel.hidden) { toggleApiPanel(false); apiStatus.focus(); }
  });
  $('#apiSave').addEventListener('click', () => {
    const v = apiUrlInput.value.trim().replace(/\/+$/, '');
    if (!v) return;
    apiBase = v;
    store.set('cardekho.api', apiBase);
    toggleApiPanel(false);
    loadOptions();
  });
  apiUrlInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); $('#apiSave').click(); } });

  /* ---------- Field errors ---------- */

  function setFieldError(name, msg) {
    const out = document.getElementById('err-' + name);
    if (out) out.textContent = msg;
    const n = document.getElementById(name); // the input, or the radiogroup for chip fields
    if (n) msg ? n.setAttribute('aria-invalid', 'true') : n.removeAttribute('aria-invalid');
  }
  const clearFieldError = (name) => setFieldError(name, '');

  function focusField(name) {
    if (name in CHOICE_FIELDS) { const r = $(`#${name} input`); if (r) r.focus(); return; }
    const n = document.getElementById(name);
    if (n && !n.disabled) n.focus();
  }

  /* ==========================================================
     Searchable dropdown (ARIA combobox with a listbox popup)
     ========================================================== */

  class Combobox {
    constructor(name, { noun, onChange }) {
      this.name = name;
      this.noun = noun;
      this.onChange = onChange;
      this.input = document.getElementById(name);
      this.list = document.getElementById(name + '-list');
      this.box = this.input.closest('.combo-box');
      this.toggle = $('.combo-toggle', this.box);
      this.hintEl = document.getElementById('hint-' + name);
      this.items = [];
      this.visible = [];
      this.value = null;
      this.active = -1;
      this.isOpen = false;
      this.bind();
    }

    bind() {
      const { input, toggle, list } = this;

      input.addEventListener('input', () => { clearFieldError(this.name); this.open(); this.render(); });
      input.addEventListener('focus', () => input.select());
      input.addEventListener('click', () => { if (!this.isOpen) { this.open(); this.render(); } });
      input.addEventListener('blur', () => { this.close(); this.commitFromText(); });

      input.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault();
          if (!this.isOpen) { this.open(); this.render(); }
          else this.move(e.key === 'ArrowDown' ? 1 : -1);
        } else if (e.key === 'Enter') {
          if (this.isOpen && this.active >= 0 && this.visible[this.active]) {
            e.preventDefault();
            this.select(this.visible[this.active]);
          }
        } else if (e.key === 'Escape' && this.isOpen) {
          e.preventDefault();
          e.stopPropagation();
          this.close();
          input.value = this.value || '';
        }
      });

      toggle.addEventListener('mousedown', (e) => e.preventDefault());
      toggle.addEventListener('click', () => {
        if (this.isOpen) { this.close(); return; }
        input.focus();
        this.open();
        this.render();
      });

      list.addEventListener('mousedown', (e) => e.preventDefault()); // keep focus in the input
      list.addEventListener('click', (e) => {
        const li = e.target.closest('li[role="option"]');
        if (li) this.select(this.visible[Number(li.dataset.i)]);
      });
      list.addEventListener('mouseover', (e) => {
        const li = e.target.closest('li[role="option"]');
        if (li) { this.active = Number(li.dataset.i); this.paintActive(false); }
      });
    }

    open() {
      this.isOpen = true;
      this.list.hidden = false;
      this.input.setAttribute('aria-expanded', 'true');
      this.box.classList.add('is-open');
    }

    close() {
      this.isOpen = false;
      this.list.hidden = true;
      this.active = -1;
      this.input.setAttribute('aria-expanded', 'false');
      this.input.removeAttribute('aria-activedescendant');
      this.box.classList.remove('is-open');
    }

    render() {
      const typed = this.input.value.trim().toLowerCase();
      // Showing the current choice should list everything, not just the choice itself
      const q = this.value && this.input.value === this.value ? '' : typed;

      let matches = this.items;
      if (q) {
        const starts = [], contains = [];
        for (const item of this.items) {
          const at = item.toLowerCase().indexOf(q);
          if (at === 0) starts.push(item);
          else if (at > 0) contains.push(item);
        }
        matches = starts.concat(contains);
      }
      this.visible = matches.slice(0, MAX_RENDERED);

      this.list.replaceChildren();
      this.visible.forEach((item, i) => {
        const li = el('li');
        li.id = `${this.name}-opt-${i}`;
        li.setAttribute('role', 'option');
        li.setAttribute('aria-selected', String(item === this.value));
        li.dataset.i = String(i);
        const at = q ? item.toLowerCase().indexOf(q) : -1;
        if (at >= 0) {
          const mark = el('mark');
          mark.textContent = item.slice(at, at + q.length);
          li.append(item.slice(0, at), mark, item.slice(at + q.length));
        } else li.textContent = item;
        this.list.append(li);
      });

      if (!matches.length) {
        const li = el('li', 'note');
        li.setAttribute('role', 'presentation');
        li.textContent = this.items.length
          ? `No ${this.noun} matches \u201C${this.input.value.trim()}\u201D.`
          : `No ${this.noun} options available.`;
        this.list.append(li);
      } else if (matches.length > MAX_RENDERED) {
        const li = el('li', 'note');
        li.setAttribute('role', 'presentation');
        li.textContent = `Showing the first ${MAX_RENDERED} of ${nf.format(matches.length)}. Keep typing to narrow the list.`;
        this.list.append(li);
      }

      const selectedAt = this.value ? this.visible.indexOf(this.value) : -1;
      this.active = q ? (this.visible.length ? 0 : -1) : selectedAt;
      this.paintActive(true);
    }

    paintActive(scroll) {
      let activeEl = null;
      $$('li[role="option"]', this.list).forEach((li, i) => {
        const on = i === this.active;
        li.classList.toggle('is-active', on);
        if (on) activeEl = li;
      });
      if (activeEl) {
        this.input.setAttribute('aria-activedescendant', activeEl.id);
        if (scroll) activeEl.scrollIntoView({ block: 'nearest' });
      } else this.input.removeAttribute('aria-activedescendant');
    }

    move(dir) {
      const n = this.visible.length;
      if (!n) return;
      this.active = this.active < 0 ? (dir > 0 ? 0 : n - 1) : (this.active + dir + n) % n;
      this.paintActive(true);
    }

    select(item) {
      const changed = item !== this.value;
      this.value = item;
      this.input.value = item;
      this.close();
      clearFieldError(this.name);
      if (changed && this.onChange) this.onChange(item);
    }

    clear(notify) {
      const had = this.value !== null;
      this.value = null;
      this.input.value = '';
      clearFieldError(this.name);
      if (notify && had && this.onChange) this.onChange(null);
    }

    // Called when focus leaves: turn typed text into a real choice, or flag it
    commitFromText() {
      const text = this.input.value.trim();
      if (!text) { if (this.value) this.clear(true); return; }
      const exact = this.items.find((i) => i.toLowerCase() === text.toLowerCase());
      if (exact) { this.select(exact); return; }
      if (this.value) { this.input.value = this.value; return; }
      setFieldError(this.name, `\u201C${text}\u201D is not in the list. Choose a ${this.noun} from it.`);
    }

    setItems(items) {
      this.items = items;
      if (this.value && !items.includes(this.value)) this.clear(true);
      if (this.isOpen) this.render();
    }

    setEnabled(on, placeholder) {
      this.input.disabled = !on;
      this.toggle.disabled = !on;
      if (placeholder !== undefined) this.input.placeholder = placeholder;
      if (!on) this.close();
    }

    setHint(text) { if (this.hintEl) this.hintEl.textContent = text; }
  }

  /* ==========================================================
     Options from the API
     ========================================================== */

  const state = { options: null };

  const boxes = {
    brand: new Combobox('brand', { noun: 'brand', onChange: () => updateModelBox() }),
    model: new Combobox('model', { noun: 'model', onChange: () => updateCarBox() }),
    car_name: new Combobox('car_name', { noun: 'car name', onChange: () => {} }),
  };

  const normKey = (k) => String(k).toLowerCase().replace(/[^a-z0-9]/g, '');

  function readOptions(raw) {
    const map = {};
    Object.entries(raw || {}).forEach(([k, v]) => { map[normKey(k)] = v; });
    const list = (...names) => {
      for (const n of names) if (Array.isArray(map[n])) return [...new Set(map[n].map(String))].sort(cmp);
      return [];
    };

    // Catalog: { brand: { model: [car names] } }, accepted only if it has exactly that shape
    let catalog = map.catalog || map.hierarchy || null;
    const isCatalog = (c) =>
      c && typeof c === 'object' && !Array.isArray(c) && Object.keys(c).length > 0 &&
      Object.values(c).every((m) => m && typeof m === 'object' && !Array.isArray(m) &&
        Object.values(m).every(Array.isArray));
    if (!isCatalog(catalog)) catalog = null;

    const opts = {
      carNames: list('carnames', 'carname', 'cars'),
      brands: list('brands', 'brand'),
      models: list('models', 'model'),
      seller: list('sellertype', 'sellertypes', 'sellers'),
      fuel: list('fueltype', 'fueltypes', 'fuel'),
      transmission: list('transmissiontype', 'transmissiontypes', 'transmission'),
      catalog,
    };
    if (catalog) {
      // Lists missing from the flat fields can be derived from the catalog the API itself sent
      if (!opts.brands.length) opts.brands = Object.keys(catalog).sort(cmp);
      if (!opts.models.length) opts.models = [...new Set(Object.values(catalog).flatMap((m) => Object.keys(m)))].sort(cmp);
      if (!opts.carNames.length) opts.carNames = [...new Set(Object.values(catalog).flatMap((m) => Object.values(m).flat().map(String)))].sort(cmp);
    }
    return opts;
  }

  const modelsFor = (brand) => {
    const c = state.options.catalog;
    return c ? Object.keys(c[brand] || {}).sort(cmp) : state.options.models;
  };
  const carsFor = (brand, model) => {
    const c = state.options.catalog;
    return c ? [...((c[brand] || {})[model] || [])].map(String).sort(cmp) : state.options.carNames;
  };

  function updateModelBox() {
    const o = state.options;
    if (!o) return;
    const { brand, model } = boxes;
    if (!o.catalog) {
      model.setItems(o.models);
      model.setEnabled(true, 'Search models');
      model.setHint('');
    } else if (!brand.value) {
      model.setItems([]);
      model.setEnabled(false, 'Choose a brand first');
      model.setHint('');
    } else {
      const list = modelsFor(brand.value);
      model.setItems(list);
      model.setEnabled(list.length > 0, list.length ? 'Search models' : 'No models listed');
      model.setHint(`${plural(list.length, 'model')} for ${brand.value}`);
    }
    updateCarBox();
  }

  function updateCarBox() {
    const o = state.options;
    if (!o) return;
    const { brand, model, car_name: car } = boxes;
    if (!o.catalog) {
      car.setItems(o.carNames);
      car.setEnabled(true, 'Search car names');
      car.setHint('');
      return;
    }
    if (!brand.value || !model.value) {
      car.setItems([]);
      car.setEnabled(false, brand.value ? 'Choose a model first' : 'Choose a brand first');
      car.setHint('');
      return;
    }
    const list = carsFor(brand.value, model.value);
    car.setItems(list);
    car.setEnabled(list.length > 0, list.length ? 'Search car names' : 'No car names listed');
    if (list.length === 1) {
      car.select(list[0]);
      car.setHint('This is the only car name for the model, so it is selected.');
    } else {
      car.setHint(`${plural(list.length, 'car name')} for ${brand.value} ${model.value}`);
    }
  }

  function buildChips(name, values) {
    const group = document.getElementById(name);
    group.classList.remove('is-loading');
    group.replaceChildren();
    if (!values.length) {
      const p = el('p', 'hint');
      p.textContent = 'The API sent no options for this field.';
      group.append(p);
      return;
    }
    values.forEach((v) => {
      const label = el('label', 'chip');
      const radio = el('input');
      radio.type = 'radio';
      radio.name = name;
      radio.value = v;
      const text = el('span');
      text.textContent = v;
      label.append(radio, text);
      group.append(label);
    });
  }
  Object.keys(CHOICE_FIELDS).forEach((name) => {
    document.getElementById(name).addEventListener('change', () => clearFieldError(name));
  });

  function setOptionsLoading() {
    Object.values(boxes).forEach((b) => { b.setEnabled(false); b.setHint(''); });
    boxes.brand.input.placeholder = 'Loading brands';
    boxes.model.input.placeholder = 'Loading models';
    boxes.car_name.input.placeholder = 'Loading car names';
    Object.keys(CHOICE_FIELDS).forEach((name) => {
      const g = document.getElementById(name);
      g.classList.add('is-loading');
      g.replaceChildren();
    });
    submitBtn.disabled = true;
    setHealth('checking');
  }

  function applyOptions(o) {
    state.options = o;
    boxes.brand.setItems(o.brands);
    boxes.brand.setEnabled(true, 'Search brands');
    boxes.brand.setHint(`${plural(o.brands.length, 'brand')} available`);
    updateModelBox();
    buildChips('seller_type', o.seller);
    buildChips('fuel_type', o.fuel);
    buildChips('transmission_type', o.transmission);
    $('#catalogNotice').hidden = !!o.catalog;
    submitBtn.disabled = false;
  }

  function describeOptionsError(err) {
    if (err.name === 'AbortError') return `No answer from ${apiBase} after ${OPTIONS_TIMEOUT_MS / 1000} seconds.`;
    if (err.kind === 'http') return `The server at ${apiBase} answered /options with status ${err.status}. Check the uvicorn terminal for the error.`;
    if (err.kind === 'json') return `${apiBase}/options did not return JSON. Check that the address points to your FastAPI app.`;
    if (err.kind === 'empty') return `/options answered, but these lists were empty or missing: ${err.missing.join(', ')}.`;
    return `Nothing answered at ${apiBase}. Check that uvicorn is running and that the API address at the top right is correct.`;
  }

  async function loadOptions() {
    $('#optionsError').hidden = true;
    setOptionsLoading();
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), OPTIONS_TIMEOUT_MS);
    try {
      let res;
      try { res = await fetch(apiBase + '/options', { signal: ctrl.signal, headers: { Accept: 'application/json' } }); }
      catch (e) { if (e.name === 'AbortError') throw e; throw Object.assign(new Error('network'), { kind: 'network' }); }
      if (!res.ok) throw Object.assign(new Error('http'), { kind: 'http', status: res.status });
      let raw;
      try { raw = await res.json(); } catch { throw Object.assign(new Error('json'), { kind: 'json' }); }

      const opts = readOptions(raw);
      const missing = [
        ['brands', opts.brands], ['models', opts.models], ['car names', opts.carNames],
        ['seller types', opts.seller], ['fuel types', opts.fuel], ['transmission types', opts.transmission],
      ].filter(([, v]) => !v.length).map(([k]) => k);
      if (missing.length) throw Object.assign(new Error('empty'), { kind: 'empty', missing });

      applyOptions(opts);
      setHealth('online');
    } catch (err) {
      if (!err.kind && err.name !== 'AbortError') console.error('Loading options failed:', err);
      state.options = null;
      $('#optionsErrorText').textContent = describeOptionsError(err);
      $('#optionsError').hidden = false;
      boxes.brand.input.placeholder = 'Options unavailable';
      boxes.model.input.placeholder = 'Options unavailable';
      boxes.car_name.input.placeholder = 'Options unavailable';
      Object.keys(CHOICE_FIELDS).forEach((name) => document.getElementById(name).classList.remove('is-loading'));
      setHealth('offline');
    } finally {
      clearTimeout(timer);
    }
  }
  $('#retryOptions').addEventListener('click', loadOptions);

  /* ---------- Number fields ---------- */

  form.addEventListener('input', (e) => {
    const name = e.target.name;
    if (!(name in NUMBER_FIELDS)) return;
    clearFieldError(name);
    if (name === 'km_driven') {
      const n = Number(e.target.value);
      $('#hint-km_driven').textContent = e.target.value !== '' && Number.isFinite(n) && n >= 1000 ? `${nf.format(n)} km` : '';
    }
  });

  /* ==========================================================
     Validation and payload
     ========================================================== */

  function validate() {
    const errors = {};
    const v = {};

    for (const [name, noun] of Object.entries(COMBO_FIELDS)) {
      const box = boxes[name];
      const typed = box.input.value.trim();
      // The text on screen must match the committed choice (Enter can submit with stale text)
      if (box.value && typed.toLowerCase() === box.value.toLowerCase()) { v[name] = box.value; continue; }
      if (name === 'model' && state.options.catalog && !boxes.brand.value) errors[name] = 'Choose a brand first.';
      else if (name === 'car_name' && state.options.catalog && !boxes.model.value) errors[name] = boxes.brand.value ? 'Choose a model first.' : 'Choose a brand first.';
      else errors[name] = typed
        ? `\u201C${typed}\u201D is not in the list. Choose a ${noun} from it.`
        : `Choose a ${noun} from the list.`;
    }

    for (const [name, noun] of Object.entries(CHOICE_FIELDS)) {
      const checked = $(`input[name="${name}"]:checked`);
      if (checked) v[name] = checked.value;
      else errors[name] = `Choose a ${noun}.`;
    }

    for (const [name, rule] of Object.entries(NUMBER_FIELDS)) {
      const raw = document.getElementById(name).value.trim();
      const n = Number(raw);
      if (raw === '') errors[name] = `Enter the ${rule.ask}.`;
      else if (!Number.isFinite(n)) errors[name] = `${rule.label} must be a number.`;
      else if (rule.int && !Number.isInteger(n)) errors[name] = `${rule.label} must be a whole number.`;
      else if (n < rule.min) errors[name] = rule.min === 0 ? `${rule.label} cannot be negative.` : `${rule.label} must be at least ${rule.min}.`;
      else v[name] = n;
    }

    const payload = {};
    PAYLOAD_FIELDS.forEach((f) => { payload[f] = v[f]; });
    return { errors, payload };
  }

  /* ==========================================================
     Odometer
     Drums spin while the model works, then slow to a stop on the
     real digits, leftmost first.
     ========================================================== */

  const odo = (() => {
    const root = $('#odo');
    const IDLE_DIGITS = 6;
    let strips = [];

    const sepAfter = (i, n) => {
      const remaining = n - i - 1;
      return remaining > 0 && (remaining === 3 || (remaining > 3 && (remaining - 3) % 2 === 0));
    };

    function build(n, mode) {
      root.replaceChildren();
      strips = [];
      root.style.setProperty('--n', String(n));
      const rupee = el('span', 'rupee');
      rupee.textContent = '\u20B9';
      root.append(rupee);

      for (let i = 0; i < n; i++) {
        const drum = el('span', 'drum');
        const strip = el('span', 'strip');
        for (let k = 0; k < 20; k++) {            // 0-9 twice, so the loop and the settle can both run forward
          const d = el('span');
          d.textContent = String(k % 10);
          strip.append(d);
        }
        if (mode === 'spin') {
          strip.classList.add('spin');
          strip.style.setProperty('--spin-d', (0.45 + Math.random() * 0.35).toFixed(2) + 's');
          strip.style.setProperty('--spin-delay', (-Math.random()).toFixed(2) + 's');
        }
        drum.append(strip);
        root.append(drum);
        strips.push(strip);
        if (sepAfter(i, n)) {
          const comma = el('span', 'sep');
          comma.textContent = ',';
          root.append(comma);
        }
      }
      root.classList.toggle('is-idle', mode === 'idle');
    }

    return {
      idle() { build(IDLE_DIGITS, 'idle'); },
      spin() { build(IDLE_DIGITS, reduceMotion ? 'idle' : 'spin'); },
      settle(digits) {
        if (strips.length !== digits.length) build(digits.length, reduceMotion ? 'idle' : 'spin');
        root.classList.remove('is-idle');

        if (reduceMotion) {
          strips.forEach((s, i) => { s.style.transform = `translateY(calc(-${10 + Number(digits[i])} * 1.25em))`; });
          return;
        }
        // Read where each drum is mid-spin before stopping its animation
        const from = strips.map((s) => {
          const t = getComputedStyle(s).transform;
          return t === 'none' ? 0 : new DOMMatrix(t).m42;
        });
        strips.forEach((s, i) => {
          s.classList.remove('spin');
          s.style.transition = 'none';
          s.style.transform = `translateY(${from[i]}px)`;
        });
        void root.offsetWidth;
        strips.forEach((s, i) => {
          s.style.transition = `transform ${1.2 + i * 0.18}s cubic-bezier(0.1, 0.7, 0.15, 1) ${i * 0.07}s`;
          s.style.transform = `translateY(calc(-${10 + Number(digits[i])} * 1.25em))`;
        });
      },
    };
  })();

  /* ==========================================================
     Prediction
     ========================================================== */

  class ApiError extends Error {
    constructor(kind, extra = {}) { super(kind); this.kind = kind; Object.assign(this, extra); }
  }

  async function requestPrediction(payload) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), PREDICT_TIMEOUT_MS);
    let res;
    try {
      res = await fetch(apiBase + '/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
        signal: ctrl.signal,
      });
    } catch (e) {
      throw e.name === 'AbortError' ? new ApiError('timeout') : new ApiError('network');
    } finally {
      clearTimeout(timer);
    }

    const data = await res.json().catch(() => null);
    if (res.status === 422) throw new ApiError('validation', { details: data && data.detail });
    if (!res.ok) throw new ApiError('server', { status: res.status, detail: data && typeof data.detail === 'string' ? data.detail : '' });
    if (!data || typeof data !== 'object') throw new ApiError('shape', { keys: [] });

    const price = data.predicted_selling_price ?? data['Predicted Selling Price'];
    if (typeof price !== 'number' || !Number.isFinite(price) || price <= 0) {
      throw new ApiError('shape', { keys: Object.keys(data), price });
    }
    return price;
  }

  function setGaugeState(s) { gauge.dataset.state = s; }

  function showPrice(price, p) {
    const rounded = Math.round(price);
    const digits = String(rounded);

    $('#priceText').textContent = `Estimated resale price ${inr.format(rounded)}, about ${priceInWords(price)}.`;
    $('#approx').textContent = `About ${priceInWords(price)}`;
    $('#carTitle').textContent = p.car_name;

    const facts = $('#facts');
    facts.replaceChildren();
    [
      p.fuel_type, p.transmission_type, p.seller_type,
      `${plural(p.vehicle_age, 'year')} old`, `${nf.format(p.km_driven)} km`,
    ].forEach((t) => { const li = el('li'); li.textContent = t; facts.append(li); });

    setGaugeState('done');
    odo.settle(digits);
  }

  const cleanMessage = (msg) => String(msg || '').replace(/^Value error,\s*/i, '');

  function showError(err) {
    const list = $('#errList');
    list.replaceChildren();
    let title = 'Something went wrong';
    let body = 'The request did not complete. Try again in a moment.';
    let canRetry = true;

    if (err.kind === 'timeout') {
      title = 'The API took too long';
      body = `No answer after ${PREDICT_TIMEOUT_MS / 1000} seconds. The model may still be loading, so try again shortly.`;
    } else if (err.kind === 'network') {
      title = 'Cannot reach the API';
      body = `Nothing answered at ${apiBase}. Check that uvicorn is running and that the API address at the top right is correct.`;
      setHealth('offline');
    } else if (err.kind === 'validation') {
      title = 'The API rejected some details';
      body = 'Fix the highlighted fields, then predict again.';
      canRetry = false;
      (Array.isArray(err.details) ? err.details : []).forEach((d) => {
        const field = d.loc && d.loc[d.loc.length - 1];
        const msg = cleanMessage(d.msg);
        const li = el('li');
        const b = el('b');
        b.textContent = FIELD_NAMES[field] || String(field);
        li.append(b, `: ${msg}`);
        list.append(li);
        if (field in FIELD_NAMES) {
          setFieldError(field, msg);
        }
      });
    } else if (err.kind === 'server') {
      title = 'The model could not produce a price';
      body = `${err.detail || 'The server returned an error.'} (status ${err.status}). The uvicorn terminal has the full traceback.`;
    } else if (err.kind === 'shape') {
      title = 'The API sent an unusable result';
      body = err.keys && err.keys.length
        ? `The response had no positive number in predicted_selling_price. It contained: ${err.keys.join(', ')}.`
        : 'The response was empty or not JSON.';
    }

    $('#errTitle').textContent = title;
    $('#errBody').textContent = body;
    $('#retryPredict').hidden = !canRetry;
    odo.idle();
    setGaugeState('error');
    const target = FOCUS_ORDER.find((n) => document.getElementById('err-' + n).textContent);
    if (target) focusField(target);
  }

  let lastPayload = null;

  async function runPrediction(payload) {
    lastPayload = payload;
    submitBtn.disabled = true;
    submitBtn.classList.add('loading');
    submitText.textContent = 'Predicting';
    setGaugeState('loading');
    odo.spin();
    if (window.matchMedia('(max-width: 999px)').matches) {
      gauge.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
    }

    const started = performance.now();
    try {
      const price = await requestPrediction(payload);
      const wait = MIN_LOADING_MS - (performance.now() - started);
      if (wait > 0) await sleep(wait);
      showPrice(price, payload);
      setHealth('online');
    } catch (err) {
      showError(err instanceof ApiError ? err : new ApiError('network'));
    } finally {
      submitBtn.disabled = false;
      submitBtn.classList.remove('loading');
      submitText.textContent = 'Predict price';
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (submitBtn.disabled || !state.options) return;
    const { errors, payload } = validate();
    [...Object.keys(COMBO_FIELDS), ...Object.keys(CHOICE_FIELDS), ...Object.keys(NUMBER_FIELDS)]
      .forEach((n) => setFieldError(n, errors[n] || ''));
    const first = FOCUS_ORDER.find((n) => errors[n]);
    if (first) { focusField(first); return; }
    runPrediction(payload);
  });

  $('#retryPredict').addEventListener('click', () => { if (lastPayload && !submitBtn.disabled) runPrediction(lastPayload); });

  $('#resetBtn').addEventListener('click', () => {
    Object.values(boxes).forEach((b) => b.clear(false));
    $$('.chips input').forEach((r) => { r.checked = false; });
    Object.keys(NUMBER_FIELDS).forEach((n) => { document.getElementById(n).value = ''; });
    $('#hint-km_driven').textContent = '';
    [...Object.keys(COMBO_FIELDS), ...Object.keys(CHOICE_FIELDS), ...Object.keys(NUMBER_FIELDS)].forEach(clearFieldError);
    boxes.brand.setHint(state.options ? `${plural(state.options.brands.length, 'brand')} available` : '');
    updateModelBox();
    $('#priceText').textContent = '';
    odo.idle();
    setGaugeState('idle');
  });

  /* ---------- Start ---------- */

  odo.idle();
  setGaugeState('idle');
  loadOptions();
})();
