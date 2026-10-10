// Reusable DOM component. Callbacks let each host supply its pattern-set editor.
function createFragmenterEditor(container, { value, editReactionPatternSet }) {
  let model;
  const clone = v => JSON.parse(JSON.stringify(v));
  const templates = {
    adduct_rules: { name: '', adduct_type: '[M+H]+', fragment_ion_adducts: ['[M+H]+'] },
    fragment_ion_adducts: '[M+H]+'
  };
  function element(tag, text) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function button(text, action) {
    const node = element('button', text); node.type = 'button'; node.onclick = action; return node;
  }
  function renderValue(parent, key, host) {
    const value = parent[key];
    const label = String(key).replaceAll('_', ' ');
    if (key === 'reaction_pattern_set') {
      const row = element('div');
      row.append(element('p', `Reaction Pattern Set: ${value.name || '(unnamed)'} · ${(value.patterns || []).length} patterns`));
      row.append(button('Edit Reaction Pattern Set', () => editReactionPatternSet(clone(value), updated => {
        parent[key] = clone(updated); render();
      })));
      host.append(row); return;
    }
    if (Array.isArray(value)) {
      const section = element('fieldset'); section.append(element('legend', label));
      value.forEach((item, index) => {
        const row = element('div'); renderValue(value, index, row);
        row.append(button('− Remove', () => { value.splice(index, 1); render(); })); section.append(row);
      });
      section.append(button('+ Add ' + (key === 'adduct_rules' ? 'AdductType' : label), () => {
        value.push(clone(templates[key] ?? (value.length ? value[0] : ''))); render();
      })); host.append(section); return;
    }
    if (value !== null && typeof value === 'object') {
      const section = element('fieldset'); section.append(element('legend', label));
      Object.keys(value).forEach(child => renderValue(value, child, section));
      host.append(section); return;
    }
    const wrapper = element('label', label); const input = element('input');
    input.type = typeof value === 'boolean' ? 'checkbox' : typeof value === 'number' ? 'number' : 'text';
    if (input.type === 'number') input.step = 'any';
    if (input.type === 'checkbox') input.checked = value; else input.value = value ?? '';
    input.oninput = () => { parent[key] = input.type === 'checkbox' ? input.checked : input.type === 'number' ? Number(input.value) : input.value; };
    wrapper.append(input); host.append(wrapper);
  }
  function render() {
    container.replaceChildren();
    Object.keys(model).forEach(key => renderValue(model, key, container));
  }
  function setValue(value) {
    value = value.fragmenter_params || value;
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Fragmenter parameters must be a JSON object.');
    model = clone(value); render();
  }
  setValue(value);
  return { getValue: () => clone(model), setValue };
}
module.exports = { createFragmenterEditor };
