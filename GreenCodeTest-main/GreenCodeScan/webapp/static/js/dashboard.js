// Dashboard functionality
document.addEventListener('DOMContentLoaded', () => {
    // Enhanced graph resizing function
    function resizeGraphs() {
        const graphs = document.querySelectorAll('.chart-container');
        graphs.forEach(graph => {
            const container = graph.closest('.dashboard-section');
            const containerWidth = container.offsetWidth;
            
            Plotly.relayout(graph, {
                width: containerWidth - 40,
                'autosize': true,
                'margin': {
                    l: 50,
                    r: 50,
                    t: 50,
                    b: 50,
                    pad: 4
                }
            });
        });
    }

    // Resize graphs on page load
    window.addEventListener('load', () => {
        setTimeout(resizeGraphs, 100);
    });

    // Debounced resize handler
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(resizeGraphs, 100);
    });

    // Observe DOM changes
    const observer = new MutationObserver(() => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(resizeGraphs, 100);
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});