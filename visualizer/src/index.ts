import createScatterplot from 'regl-scatterplot';
import * as d3 from 'd3';
import { html, render } from 'lit-html';
import * as duckdb from '@duckdb/duckdb-wasm';

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
const COLOR_GRADIENT = [
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

let db: duckdb.AsyncDuckDB;

async function initDuckDB() {
    const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();
    
    const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);
    const worker_url = URL.createObjectURL(
        new Blob([`importScripts("${bundle.mainWorker}");`], { type: 'text/javascript' })
    );

    const worker = new Worker(worker_url);
    const logger = new duckdb.ConsoleLogger();
    db = new duckdb.AsyncDuckDB(logger, worker);
    await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
    URL.revokeObjectURL(worker_url);
}

async function loadParquetWithDuckDB(filename: string, query: string) {
    const response = await fetch(`/viz_data/${filename}`);
    if (!response.ok) throw new Error(`Failed to load ${filename}`);
    const buffer = await response.arrayBuffer();
    
    const conn = await db.connect();
    await db.registerFileBuffer(filename, new Uint8Array(buffer));
    const result = await conn.query(query);
    await conn.close();
    
    return result;
}

async function init() {
    try {
        starColorScale = d3.scaleLinear<string>()
            .domain(STAR_DOMAIN)
            .range(COLOR_GRADIENT)
            .clamp(true);

        smoothStarGradient = d3.quantize(t => starColorScale(t * 8), 256).map(c => d3.color(c)!.formatHex());

        await initDuckDB();

        const [pointsResult, attributesResult, neighborsResult, metaResult, statusMapResult] = await Promise.all([
            loadParquetWithDuckDB('points.parquet', 'SELECT * FROM "points.parquet"'),
            loadParquetWithDuckDB('attributes.parquet', 'SELECT * FROM "attributes.parquet"'),
            loadParquetWithDuckDB('neighbors.parquet', 'SELECT * FROM "neighbors.parquet"'),
            loadParquetWithDuckDB('meta.parquet', 'SELECT * FROM "meta.parquet"'),
            loadParquetWithDuckDB('status_map.parquet', 'SELECT * FROM "status_map.parquet"')
        ]);

        const ids = pointsResult.toArray().map((row: any) => Number(row.id));
        const x = pointsResult.toArray().map((row: any) => Number(row.x));
        const y = pointsResult.toArray().map((row: any) => Number(row.y));

        const attrRows = attributesResult.toArray();
        const titles = attrRows.map((row: any) => row.title);
        const artists = attrRows.map((row: any) => row.artist);
        const mappers = attrRows.map((row: any) => row.mapper);
        const diffs = attrRows.map((row: any) => row.diff);
        const stars = attrRows.map((row: any) => Number(row.stars));
        const dates = attrRows.map((row: any) => row.date);
        const playcounts = attrRows.map((row: any) => Number(row.playcount));
        const max_combos = attrRows.map((row: any) => Number(row.max_combo));
        const lengths = attrRows.map((row: any) => Number(row.length));
        const bpms = attrRows.map((row: any) => Number(row.bpm));
        const statuses = attrRows.map((row: any) => row.status);

        const neighborRows = neighborsResult.toArray();
        const neighbor_indices = neighborRows.map((row: any) => 
            Array.from(row.indices).map((v: any) => Number(v))
        );
        const neighbor_distances = neighborRows.map((row: any) => 
            Array.from(row.distances).map((v: any) => Number(v))
        );

        const status_map: Record<string, string> = {};
        const statusMapRows = statusMapResult.toArray();
        for (const row of statusMapRows) {
            status_map[row.status_code] = row.status_name;
        }

        globalData = {
            ids,
            x,
            y,
            titles,
            artists,
            mappers,
            diffs,
            stars,
            bpms,
            lengths,
            max_combos,
            dates,
            statuses,
            neighbor_indices,
            neighbor_distances
        };

        meta = { status_map };

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
    const colorSelect = document.getElementById('color-mode') as HTMLSelectElement;
    if (colorSelect) {
        colorSelect.value = 'stars';
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

    const canvasWidth = canvasWrapper.clientWidth;
    const canvasHeight = canvasWrapper.clientHeight;

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
        pointColor: COLOR_GRADIENT,
    });

    const resizeObserver = new ResizeObserver((entries: ResizeObserverEntry[]) => {
        for (const entry of entries) {
            scatterplot.set({ width: entry.contentRect.width, height: entry.contentRect.height });
        }
    });
    resizeObserver.observe(canvasWrapper);

    scatterplot.subscribe('pointover', (pointIndex: number) => {
        if (globalData) {
            tooltip.style.display = 'block';
            render(html`${globalData.titles[pointIndex]} [${globalData.diffs[pointIndex]}]`, tooltip);
            canvasWrapper.style.cursor = 'pointer';
        }
    });

    scatterplot.subscribe('pointout', () => {
        tooltip.style.display = 'none';
        canvasWrapper.style.cursor = 'default';
    });

    canvasWrapper.addEventListener('mousemove', (e) => {
        tooltip.style.left = (e.clientX + 15) + 'px';
        tooltip.style.top = (e.clientY + 15) + 'px';
    });

    canvasWrapper.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
        canvasWrapper.style.cursor = 'default';
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

    if (!globalData) return;

    let [xMin, xMax] = d3.extent(globalData.x) as [number, number];
    let [yMin, yMax] = d3.extent(globalData.y) as [number, number];

    const dataAspect = (xMax - xMin) / (yMax - yMin);
    const screenAspect = canvasWrapper.clientWidth / canvasWrapper.clientHeight;

    if ((dataAspect < 1 && screenAspect > 1) || (dataAspect > 1 && screenAspect < 1)) {
        [globalData.x, globalData.y] = [globalData.y, globalData.x.map(v => -v)];
        
        [xMin, xMax, yMin, yMax] = [yMin, yMax, -xMax, -xMin];
    }

    updateColorMode('stars');

    const padding = 0.05;
    const width = xMax - xMin;
    const height = yMax - yMin;

    scatterplot.zoomToArea({
        x: xMin - width * padding,
        y: yMin - height * padding,
        width: width * (1 + padding * 2),
        height: height * (1 + padding * 2)
    }, { transition: false });
}

function formatLength(seconds: number): string {
    return new Date(seconds * 1000).toISOString().slice(14, 19);
}

function formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleDateString();
}

function generateColorValues(mode: string): number[] {
    if (!globalData) return [];

    if (mode === 'status') {
        const statusMap: Record<string, number> = {
            "1": 0, "2": 1, "3": 2, "4": 3,
            "0": 4, "-1": 5, "-2": 6
        };
        return Array.from(globalData.statuses, s => statusMap[String(s)] ?? 0);
    }

    let values: number[] = [];
    let scale = d3.scaleLinear().range([0, 1]).clamp(true);

    if (mode === 'stars') {
        values = globalData.stars;
        scale.domain([0, 8]);
    } 
    else {
        let rawValues: any[];
        
        switch(mode) {
            case 'bpm': rawValues = globalData.bpms; break;
            case 'date': rawValues = globalData.dates; break; 
            case 'length': rawValues = globalData.lengths; break;
            case 'maxcombo': rawValues = globalData.max_combos; break;
            default: throw new Error(`Unsupported: ${mode}`);
        }

        values = mode === 'date' 
            ? rawValues.map(d => new Date(d).getTime()) 
            : rawValues.map(Number);

        const sorted = values.sort(); 
        const p10 = d3.quantileSorted(sorted, 0.25) || 0;
        const p90 = d3.quantileSorted(sorted, 0.75) || 1;
        
        scale.domain([p10, p90]);
    }

    return values.map(v => scale(v));
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
    } else if (mode === 'status') {
        colorMap = Object.values(STATUS_COLORS);
    } else {
        const interpolator = d3.interpolateRgbBasis(COLOR_GRADIENT);
        colorMap = d3.quantize(interpolator, 256).map(c => d3.color(c)?.formatHex() || "#000000");
    }
    scatterplot.set({ pointColor: colorMap });

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
    const externalUrl = `https://osu.ppy.sh/b/${globalData.ids[idx]}`;

    return html`
        <div class="neighbor-item" @click=${() => selectBeatmap(idx)}>
            <div class="neighbor-title">
                <span class="neighbor-title-text" title="${globalData.titles[idx] as string}">${globalData.titles[idx]}</span>
                <a class="mini-external-link" href="${externalUrl}" target="_blank" @click=${(e: Event) => e.stopPropagation()} title="Open in osu!">↗</a>
                <span class="neighbor-title-diff" title="${globalData.diffs[idx] as string}">${globalData.diffs[idx]}</span>
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

    const infoItems = [
        { label: 'Title', val: globalData.titles[idx], title: true },
        { label: 'Artist', val: globalData.artists[idx], title: true },
        { label: 'Mapper', val: globalData.mappers[idx] },
        { label: 'Version', val: globalData.diffs[idx], title: true },
        { label: 'Stars', val: `${globalData.stars[idx]} ★` },
        { label: 'BPM', val: `${globalData.bpms[idx]} BPM` },
        { label: 'Length', val: formatLength(globalData.lengths[idx] as number) },
        { label: 'Max Combo', val: `${globalData.max_combos[idx]}x` },
        { label: 'Date', val: formatDate(globalData.dates[idx] as string) },
        { label: 'Status', val: statusTxt },
    ];

    return html`${infoItems.map(item => html`
        <div class="info-row">
            <span class="info-label">${item.label}</span>
            <span class="info-val" title="${item.title ? item.val as string : ''}">${item.val}</span>
        </div>
    `)}`;
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
