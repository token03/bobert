import createScatterplot from 'regl-scatterplot';
import * as d3 from 'd3';
import { html, render } from 'lit-html';

interface VizData {
    ids: number[];
    x: number[];
    y: number[];
    titles: string[];
    artists: string[];
    mappers: string[];
    diffs: string[];
    stars: number[];
    bpms: number[];
    lengths: number[];
    max_combos: number[];
    dates: string[];
    statuses: number[];
    neighbor_indices: number[][];
    neighbor_distances: number[][];
}

interface MetaData {
    status_map: Record<string, string>;
}

interface AppState {
    viewMode: 'empty' | 'single' | 'multi';
    selectedIdx: number | null;
    multiIndices: number[];
}

const STATUS_COLORS: Record<string, string> = {
    "1": "#2ea4ff", "2": "#a5dc42", "3": "#55ccff", "4": "#ff66aa",
    "0": "#ffd966", "-1": "#ffcc22", "-2": "#666666"
};

const STAR_DOMAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const STAR_RANGE = [
    '#4290fb', '#4fc0ff', '#4fffd5', '#7cff4f',
    '#f6f05c', '#ff8068', '#ff3c71', '#6563de', '#18158e'
];

const SIZE_BASE = 4;
const SIZE_SELECTED = 8;
const SIZE_NEIGHBOR = 8;

let scatterplot: any; 
let globalData: VizData | null = null;
let meta: MetaData | null = null;
let idMap = new Map<string, number>();

const appState: AppState = {
    viewMode: 'empty',
    selectedIdx: null,
    multiIndices: []
};

let starColorScale: d3.ScaleLinear<string, string>;
let smoothStarGradient: string[] = [];

async function init() {
    try {
        starColorScale = d3.scaleLinear<string>()
            .domain(STAR_DOMAIN)
            .range(STAR_RANGE)
            .clamp(true);

        smoothStarGradient = d3.quantize(t => starColorScale(t * 8), 256).map(c => d3.color(c)!.formatHex());

        const response = await fetch('/viz_data.json');
        if (!response.ok) throw new Error("Failed to load viz_data.json");

        const payload = await response.json();
        globalData = payload.data as VizData;
        meta = payload.meta as MetaData;

        globalData.ids.forEach((id: number, index: number) => {
            idMap.set(String(id), index);
        });

        const loader = document.getElementById('loader');
        if (loader) loader.style.display = 'none';

        initScatterplot();
        initUI();

    } catch (e: any) {
        const loader = document.getElementById('loader');
        if (loader) {
            render(html`<span class="error-message">ERROR: ${e.message}</span>`, loader);
        }
        console.error(e);
    }
}

function initUI() {
    const colorSelect = document.getElementById('color-mode');
    if (colorSelect) {
        colorSelect.addEventListener('change', (e) => {
            const target = e.target as HTMLSelectElement;
            updateColorMode(target.value);
        });
    }

    const searchBtn = document.getElementById('search-btn');
    if (searchBtn) searchBtn.addEventListener('click', doSearch);

    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('keydown', (e) => {
            if ((e as KeyboardEvent).key === 'Enter') doSearch();
        });
    }

    renderPanel();
}

function initScatterplot() {
    const canvasWrapper = document.getElementById('chart-canvas-wrapper');
    const tooltip = document.getElementById('hover-tooltip');

    if (!canvasWrapper || !tooltip) return;

    tooltip.style.pointerEvents = 'none';

    const canvas = document.createElement('canvas');
    canvasWrapper.appendChild(canvas);

    scatterplot = createScatterplot({
        canvas: canvas,
        width: canvasWrapper.clientWidth,
        height: canvasWrapper.clientHeight,
        pointSize: [SIZE_BASE, SIZE_NEIGHBOR, SIZE_SELECTED],
        opacity: 0.8,
        backgroundColor: '#0f0f12',
        lassoInitiator: false,
        colorBy: 'valueA',
        sizeBy: 'valueB',
        pointColor: STAR_RANGE,
    });

    const resizeObserver = new ResizeObserver((entries: ResizeObserverEntry[]) => {
        for (const entry of entries) {
            scatterplot.set({ width: entry.contentRect.width, height: entry.contentRect.height });
        }
    });
    resizeObserver.observe(canvasWrapper);

    let lastHoverPoint: number | null = null;
    let hoverUpdateTime = 0;

    scatterplot.subscribe('pointover', (pointIndex: number | null) => {
        lastHoverPoint = pointIndex;
        hoverUpdateTime = Date.now();

        if (pointIndex !== null && globalData) {
            tooltip.style.display = 'block';
            render(html`${globalData.titles[pointIndex]} [${globalData.diffs[pointIndex]}]`, tooltip);
            canvasWrapper.style.cursor = 'pointer';
        } else {
            tooltip.style.display = 'none';
            canvasWrapper.style.cursor = 'default';
        }
    });

    canvasWrapper.addEventListener('mousemove', (e) => {
        tooltip.style.left = (e.clientX + 15) + 'px';
        tooltip.style.top = (e.clientY + 15) + 'px';

        if (lastHoverPoint !== null && Date.now() - hoverUpdateTime > 50) {
            tooltip.style.display = 'none';
            canvasWrapper.style.cursor = 'default';
            lastHoverPoint = null;
        }
    });

    canvasWrapper.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
        canvasWrapper.style.cursor = 'default';
        lastHoverPoint = null;
    });

    scatterplot.subscribe('pointclick', (pointIndex: number | null) => {
        if (pointIndex !== null) selectBeatmap(pointIndex);
    });

    scatterplot.subscribe('select', ({ points }: { points: number[] }) => {
        if (points.length === 0) {
            showEmptyState();
        } else if (points.length === 1) {
            selectBeatmap(points[0] as number);
        } else {
            showLassoSelection(points);
        }
    });

    updateColorMode('stars');
}

function formatLength(seconds: number): string {
    return new Date(seconds * 1000).toISOString().slice(14, 19);
}

function formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleDateString();
}

function generateColorValues(mode: string): number[] {
    if (!globalData) return [];

    const count = globalData.x.length;
    const values = new Array(count);

    if (mode === 'stars') {
        for (let i = 0; i < count; i++) {
            values[i] = Math.min(1, Math.max(0, (globalData.stars[i] || 0) / 8));
        }
    } else if (mode === 'status') {
        const statusMap: Record<string, number> = {
            "1": 0, "2": 1, "3": 2, "4": 3,
            "0": 4, "-1": 5, "-2": 6
        };
        for (let i = 0; i < count; i++) {
            values[i] = statusMap[String(globalData.statuses[i])] ?? 0;
        }
    } else if (mode === 'bpm' || mode === 'date' || mode === 'length' || mode === 'maxcombo') {
        let dataKey: keyof VizData;
        switch(mode) {
            case 'bpm': dataKey = 'bpms'; break;
            case 'date': dataKey = 'dates'; break;
            case 'length': dataKey = 'lengths'; break;
            default: dataKey = 'max_combos'; break;
        }

        const rawValues = globalData[dataKey] as any[];
        let numericValues: number[];

        if (mode === 'date') {
            numericValues = rawValues.map((d: string) => new Date(d).getTime());
        } else {
            numericValues = rawValues.map((v: any) => +v || 0);
        }

        const validValues = numericValues.filter((v) => !isNaN(v) && isFinite(v));
        if (validValues.length === 0) throw new Error(`No valid values for: ${mode}`);

        const p10 = d3.quantile(validValues, 0.25)!;
        const p90 = d3.quantile(validValues, 0.75)!;
        const range = p90 - p10;

        if (range === 0) {
            values.fill(0.5);
        } else {
            for (let i = 0; i < count; i++) {
                const val = numericValues[i];
                const clamped = Math.min(p90, Math.max(p10, val as number));
                values[i] = (clamped - p10) / range;
            }
        }
    } else {
        throw new Error(`Unsupported color mode: ${mode}`);
    }

    return values;
}

function generateSizeValues(highlightIdx: number | null = null, neighborIndices: number[] = []): number[] {
    if (!globalData) return [];

    const count = globalData.x.length;
    const values = new Array(count).fill(0);

    neighborIndices.forEach((nIdx) => values[nIdx] = 1);
    if (highlightIdx !== null) values[highlightIdx] = 2;

    return values;
}

function updateColorMode(mode: string) {
    if (!scatterplot || !globalData) return;

    const colorValues = generateColorValues(mode);
    let sizeValues: number[] = [];
    if (appState.selectedIdx !== null) {
        sizeValues = generateSizeValues(appState.selectedIdx, globalData.neighbor_indices[appState.selectedIdx]);
    } else {
        sizeValues = generateSizeValues();
    }

    let colorMap: string[] = [];
    if (mode === 'stars') {
        colorMap = smoothStarGradient;
        scatterplot.set({ pointColor: colorMap });
    } else if (mode === 'status') {
        colorMap = Object.values(STATUS_COLORS);
        scatterplot.set({ pointColor: colorMap });
    } else {
        const darkColor = '#a6084f';
        const lightColor = '#ff87c6';
        const interpolator = d3.interpolateRgb(darkColor, lightColor);
        colorMap = d3.quantize(interpolator, 256).map(c => d3.color(c)?.formatHex() || "#000000");
        scatterplot.set({ pointColor: colorMap });
    }

    scatterplot.draw({
        x: globalData.x,
        y: globalData.y,
        valueA: colorValues,
        valueB: sizeValues
    });
}

function extractBeatmapId(input: string): string {
    const trimmed = input.trim();
    if (trimmed.includes('osu.ppy.sh')) {
        const hashMatch = trimmed.match(/beatmapsets\/\d+#(?:osu|taiko|fruits|mania)\/(\d+)/);
        if (hashMatch) return hashMatch[1] as string;
        const beatmapsMatch = trimmed.match(/\/(?:beatmaps|b)\/(\d+)/);
        if (beatmapsMatch) return beatmapsMatch[1] as string;
    }
    return trimmed;
}

function doSearch() {
    const input = document.getElementById('search-input') as HTMLInputElement;
    if (!input) return;
    
    const rawInput = input.value.trim();
    const id = extractBeatmapId(rawInput);
    const errorMsg = document.getElementById('search-error');

    if (idMap.has(id)) {
        if (errorMsg) errorMsg.style.display = 'none';
        selectBeatmap(idMap.get(id)!);
    } else {
        if (errorMsg) errorMsg.style.display = 'block';
    }
}

const beatmapItemTemplate = (idx: number, similarity?: number) => {
    if (!globalData) return html``;
    
    const simString = similarity !== undefined ? (similarity * 100).toFixed(2) + '%' : '';

    return html`
        <div class="neighbor-item" @click=${() => selectBeatmap(idx)}>
            <div class="neighbor-title">
                <span class="neighbor-title-text" title="${globalData.titles[idx]}">${globalData.titles[idx]}</span>
                <span class="neighbor-title-diff" title="${globalData.diffs[idx]}">${globalData.diffs[idx]}</span>
            </div>
            <div class="neighbor-sub">
                <span>${globalData.stars[idx]} ★ · ${globalData.bpms[idx]} BPM · ${formatLength(globalData.lengths[idx] as number)}</span>
                ${similarity !== undefined ? html`<span class="similarity-score">${simString}</span>` : ''}
            </div>
        </div>
    `;
};

const metaInfoTemplate = (idx: number) => {
    if (!globalData || !meta) return html``;
    
    const statusTxt = meta.status_map[globalData.statuses[idx] as number] || "Unknown";
    const externalUrl = `https://osu.ppy.sh/b/${globalData.ids[idx]}`;

    const extLinkEl = document.getElementById('external-link') as HTMLAnchorElement;
    if (extLinkEl) extLinkEl.href = externalUrl;

    return html`
        <div class="info-row"><span class="info-label">Title</span> <span class="info-val" title="${globalData.titles[idx]}">${globalData.titles[idx]}</span></div>
        <div class="info-row"><span class="info-label">Artist</span> <span class="info-val" title="${globalData.artists[idx]}">${globalData.artists[idx]}</span></div>
        <div class="info-row"><span class="info-label">Mapper</span> <span class="info-val">${globalData.mappers[idx]}</span></div>
        <div class="info-row"><span class="info-label">Diff</span> <span class="info-val" title="${globalData.diffs[idx]}">${globalData.diffs[idx]}</span></div>
        <div class="info-row"><span class="info-label">Stars</span> <span class="info-val">${globalData.stars[idx]} ★</span></div>
        <div class="info-row"><span class="info-label">BPM</span> <span class="info-val">${globalData.bpms[idx]} BPM</span></div>
        <div class="info-row"><span class="info-label">Length</span> <span class="info-val">${formatLength(globalData.lengths[idx] as number)}</span></div>
        <div class="info-row"><span class="info-label">Max Combo</span> <span class="info-val">${globalData.max_combos[idx]}x</span></div>
        <div class="info-row"><span class="info-label">Date</span> <span class="info-val">${formatDate(globalData.dates[idx] as string)}</span></div>
        <div class="info-row"><span class="info-label">Status</span> <span class="info-val">${statusTxt}</span></div>
    `;
};

function renderPanel() {
    const selectionPanel = document.getElementById('selection-panel');
    const emptyState = document.getElementById('empty-state');
    const singlePanel = document.getElementById('single-select-panel');
    const multiPanel = document.getElementById('multi-select-panel');

    if (!selectionPanel || !emptyState || !singlePanel || !multiPanel) return;

    selectionPanel.style.display = 'block';
    emptyState.style.display = appState.viewMode === 'empty' ? 'block' : 'none';
    singlePanel.style.display = appState.viewMode === 'single' ? 'block' : 'none';
    multiPanel.style.display = appState.viewMode === 'multi' ? 'block' : 'none';

    if (appState.viewMode === 'single' && appState.selectedIdx !== null && globalData) {
        render(metaInfoTemplate(appState.selectedIdx), document.getElementById('meta-container')!);
        
        const neighbors = globalData.neighbor_indices[appState.selectedIdx];
        const distances = globalData.neighbor_distances[appState.selectedIdx];

        if (!neighbors || !distances) {
            render(html`<div>No neighbor data available.</div>`, document.getElementById('neighbor-container')!);
            return;
        }
        
        const neighborTemplates = neighbors.map((nIdx, i) => {
            const similarity = 1 - (distances[i] as number);
            return beatmapItemTemplate(nIdx, similarity);
        });
        
        render(html`${neighborTemplates}`, document.getElementById('neighbor-container')!);
    } 
    else if (appState.viewMode === 'multi') {
        const countEl = document.getElementById('select-count');
        if (countEl) countEl.textContent = String(appState.multiIndices.length);

        const listTemplates = appState.multiIndices.map(idx => beatmapItemTemplate(idx));
        render(html`${listTemplates}`, document.getElementById('multi-select-container')!);
    }
}

function showEmptyState() {
    appState.viewMode = 'empty';
    appState.selectedIdx = null;
    appState.multiIndices = [];
    
    renderPanel();
    
    if (scatterplot && globalData) {
        const colorMode = (document.getElementById('color-mode') as HTMLSelectElement).value;
        const colorValues = generateColorValues(colorMode);
        const sizeValues = generateSizeValues();
        scatterplot.draw({ 
            x: globalData.x, y: globalData.y, 
            valueA: colorValues, valueB: sizeValues 
        });
    }
}

function showLassoSelection(indices: number[]) {
    appState.viewMode = 'multi';
    appState.selectedIdx = null;
    appState.multiIndices = indices;

    renderPanel();

    if (scatterplot && globalData) {
        const colorMode = (document.getElementById('color-mode') as HTMLSelectElement).value;
        const colorValues = generateColorValues(colorMode);
        const sizeValues = generateSizeValues(null, indices);
        scatterplot.draw({ 
            x: globalData.x, y: globalData.y, 
            valueA: colorValues, valueB: sizeValues 
        });
    }
}

async function selectBeatmap(idx: number) {
    if (!globalData) return;

    appState.viewMode = 'single';
    appState.selectedIdx = idx;
    appState.multiIndices = [];

    renderPanel();

    const neighborIndices = globalData.neighbor_indices[idx] || [];
    const sizeValues = generateSizeValues(idx, neighborIndices);

    const colorMode = (document.getElementById('color-mode') as HTMLSelectElement).value;
    const colorValues = generateColorValues(colorMode);
    
    await scatterplot.draw({ 
        x: globalData.x, 
        y: globalData.y, 
        valueA: colorValues, 
        valueB: sizeValues
    });

    scatterplot.select([idx]);
    
    scatterplot.zoomToLocation(
        [globalData.x[idx], globalData.y[idx]], 
        0.5, 
        { transition: true, duration: 800 }
    );
}

(window as any).selectBeatmap = selectBeatmap;

init();