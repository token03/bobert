import createScatterplot from 'regl-scatterplot';
import * as d3 from 'd3';

let scatterplot: any;
let globalData: any = null;
let meta: any = null;
let idMap = new Map<string, number>();

const STATUS_COLORS: Record<string, string> = {
    "1": "#2ea4ff", "2": "#a5dc42", "3": "#55ccff", "4": "#ff66aa",
    "0": "#ffd966", "-1": "#ffcc22", "-2": "#666666"
};

const STAR_DOMAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const STAR_RANGE = [
    '#4290fb', '#4fc0ff', '#4fffd5', '#7cff4f',
    '#f6f05c', '#ff8068', '#ff3c71', '#6563de', '#18158e'
];

let starColorScale: d3.ScaleLinear<string, string>;
let smoothStarGradient: string[] = [];

async function init() {
    try {
        starColorScale = d3.scaleLinear<string>()
            .domain(STAR_DOMAIN)
            .range(STAR_RANGE)
            .clamp(true);

        const GRADIENT_STEPS = 256;
        smoothStarGradient = new Array(GRADIENT_STEPS).fill(0).map((_, i) => {
            const t = i / (GRADIENT_STEPS - 1); 
            const colorStr = starColorScale(t * 8); 
            return d3.color(colorStr)?.formatHex() || "#000000";
        });

        const response = await fetch('/viz_data.json');
        if (!response.ok) throw new Error("Failed to load viz_data.json");

        const payload = await response.json();
        globalData = payload.data;
        meta = payload.meta;

        globalData.ids.forEach((id: number, index: number) => {
            idMap.set(String(id), index);
        });

        document.getElementById('loader')!.style.display = 'none';

        initScatterplot();
        
        document.getElementById('color-mode')!.addEventListener('change', (e: any) => updateColorMode(e.target.value));
        document.getElementById('search-btn')!.addEventListener('click', doSearch);
        document.getElementById('search-input')!.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') doSearch();
        });

    } catch (e: any) {
        const loader = document.getElementById('loader');
        if (loader) {
            loader.innerHTML = `<span style="color:red; font-size: 1rem; text-align:center;">ERROR: ${e.message}</span>`;
        }
        console.error(e);
    }
}

function initScatterplot() {
    const canvasWrapper = document.getElementById('chart-canvas-wrapper')!;
    const tooltip = document.getElementById('hover-tooltip')!;

    tooltip.style.pointerEvents = 'none'; 

    const canvas = document.createElement('canvas');
    canvasWrapper.appendChild(canvas);

    scatterplot = createScatterplot({
        canvas: canvas,
        width: canvasWrapper.clientWidth,
        height: canvasWrapper.clientHeight,
        pointSize: 4,
        opacity: 0.8,
        backgroundColor: '#0f0f12',
        lassoInitiator: false,
        cameraRotation: false,
        colorBy: 'valueA',
        pointColor: STAR_RANGE,  
    });
    
    const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
            scatterplot.set({ width: entry.contentRect.width, height: entry.contentRect.height });
        }
    });
    resizeObserver.observe(canvasWrapper);

    let lastHoverPoint: number | null = null;
    let hoverUpdateTime = 0;

    scatterplot.subscribe('pointover', (pointIndex: number | null) => {
        lastHoverPoint = pointIndex;
        hoverUpdateTime = Date.now();
        
        if (pointIndex !== null) {
            tooltip.style.display = 'block';
            tooltip.innerHTML = `${globalData.titles[pointIndex]} [${globalData.diffs[pointIndex]}]`;
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
            selectBeatmap(points[0]);
        } else {
            showLassoSelection(points);
        }
    });

    updateColorMode('stars');
}

function generateColorValues(mode: string): number[] {
    if (!globalData) return [];
    
    const count = globalData.x.length;
    const values = new Array(count);

    if (mode === 'stars') {
        for (let i = 0; i < count; i++) {
            const starValue = globalData.stars[i];
            const starNum = +starValue || 0;
            values[i] = Math.min(1, Math.max(0, starNum / 8));
        }
    } else if (mode === 'status') {
        for (let i = 0; i < count; i++) {
            const statusStr = String(globalData.statuses[i]);
            const statusMap: Record<string, number> = {
                "1": 0, "2": 1, "3": 2, "4": 3,
                "0": 4, "-1": 5, "-2": 6
            };
            values[i] = statusMap[statusStr] ?? 0;
        }
        
    } else {
        values.fill(0.5);
    }
    
    return values;
}

function updateColorMode(mode: string) {
    if (!scatterplot || !globalData) return;
    
    const colorValues = generateColorValues(mode);
    
    let colorMap: string[] = [];
    if (mode === 'stars') {
        colorMap = smoothStarGradient; 
        scatterplot.set({ pointColor: colorMap });
    } else if (mode === 'status') {
        colorMap = Object.values(STATUS_COLORS);
        scatterplot.set({ pointColor: colorMap });
    } else {
        colorMap = ['#5c7cfa'];
        scatterplot.set({ pointColor: colorMap });
    }
    
    scatterplot.draw({
        x: globalData.x,
        y: globalData.y,
        valueA: colorValues  
    });
}

function doSearch() {
    const input = document.getElementById('search-input') as HTMLInputElement;
    const id = input.value.trim();
    const errorMsg = document.getElementById('search-error')!;

    if (idMap.has(id)) {
        errorMsg.style.display = 'none';
        selectBeatmap(idMap.get(id)!);
    } else {
        errorMsg.style.display = 'block';
    }
}

(window as any).selectBeatmap = selectBeatmap;

function showEmptyState() {
    const selectionPanel = document.getElementById('selection-panel')!;
    selectionPanel.style.display = 'block';
    document.getElementById('empty-state')!.style.display = 'block';
    document.getElementById('single-select-panel')!.style.display = 'none';
    document.getElementById('multi-select-panel')!.style.display = 'none';
}

function showLassoSelection(indices: number[]) {
    if (!globalData) return;
    
    const selectionPanel = document.getElementById('selection-panel')!;
    selectionPanel.style.display = 'block';
    document.getElementById('empty-state')!.style.display = 'none';
    document.getElementById('single-select-panel')!.style.display = 'none';
    
    const multiPanel = document.getElementById('multi-select-panel')!;
    multiPanel.style.display = 'block';
    document.getElementById('select-count')!.textContent = String(indices.length);
    
    let html = '';
    indices.forEach((idx: number) => {
        html += `
            <div class="neighbor-item" onclick="selectBeatmap(${idx})">
                <div class="neighbor-title">
                    <span class="neighbor-title-text">${globalData.titles[idx]}</span>
                    <span class="neighbor-title-diff">${globalData.diffs[idx]}</span>
                </div>
                <div class="neighbor-sub">
                    <span>${globalData.stars[idx]} ★</span>
                </div>
            </div>
        `;
    });
    document.getElementById('multi-select-container')!.innerHTML = html;
}

function selectBeatmap(idx: number) {
    if (!globalData) return;

    const selectionPanel = document.getElementById('selection-panel')!;
    selectionPanel.style.display = 'block';
    document.getElementById('empty-state')!.style.display = 'none';
    document.getElementById('multi-select-panel')!.style.display = 'none';
    
    const singlePanel = document.getElementById('single-select-panel')!;
    singlePanel.style.display = 'block';

    const statusTxt = meta.status_map[globalData.statuses[idx]] || "Unknown";
    
    document.getElementById('meta-container')!.innerHTML = `
        <div class="info-row"><span class="info-label">Title</span> <span class="info-val" title="${globalData.titles[idx]}">${globalData.titles[idx]}</span></div>
        <div class="info-row"><span class="info-label">Artist</span> <span class="info-val" title="${globalData.artists[idx]}">${globalData.artists[idx]}</span></div>
        <div class="info-row"><span class="info-label">Mapper</span> <span class="info-val">${globalData.mappers[idx]}</span></div>
        <div class="info-row"><span class="info-label">Diff</span> <span class="info-val">${globalData.diffs[idx]}</span></div>
        <div class="info-row"><span class="info-label">Stars</span> <span class="info-val">${globalData.stars[idx]} ★</span></div>
        <div class="info-row"><span class="info-label">Status</span> <span class="info-val">${statusTxt}</span></div>
        <div class="info-row"><span class="info-label">Link</span> <span class="info-val"><a href="https://osu.ppy.sh/b/${globalData.ids[idx]}" target="_blank" style="color:#5c7cfa">Open in osu!</a></span></div>
    `;

    let nbrHtml = '';
    globalData.neighbor_indices[idx].forEach((nIdx: number, i: number) => {
        // Calculate cosine similarity from cosine distance
        const cosineDist = globalData.neighbor_distances[idx][i];
        const similarity = (1 - cosineDist) * 100;
        const simStr = similarity.toFixed(2) + '%';
        
        nbrHtml += `
            <div class="neighbor-item" onclick="selectBeatmap(${nIdx})">
                <div class="neighbor-title">
                    <span class="neighbor-title-text">${globalData.titles[nIdx]}</span>
                    <span class="neighbor-title-diff">${globalData.diffs[nIdx]}</span>
                </div>
                <div class="neighbor-sub">
                    <span>${globalData.stars[nIdx]} ★</span>
                    <span style="color: #5c7cfa; font-weight: 600;">${simStr}</span>
                </div>
            </div>
        `;
    });
    document.getElementById('neighbor-container')!.innerHTML = nbrHtml;

    scatterplot.select([idx]);
    
    scatterplot.zoomToLocation(
        [globalData.x[idx], globalData.y[idx]], 
        0.5, 
        { transition: true, duration: 800 }
    );
}

init();