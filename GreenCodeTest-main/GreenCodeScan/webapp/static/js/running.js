// Running page functionality
document.addEventListener('DOMContentLoaded', () => {
    let currentProgress = 0;
    const progressFill = document.querySelector('.progress-fill');
    const progressPercentage = document.querySelector('.progress-percentage');
    const timerOverlay = document.getElementById('timerOverlay');
    const timerCountdown = document.getElementById('timerCountdown');
    const timerProgress = document.querySelector('.timer-progress');
    
    function updateProgress(stage) {
        const progress = stage * 33.33;
        progressFill.style.width = `${progress}%`;
        progressPercentage.textContent = `${Math.round(progress)}%`;
        
        if (progress >= 100) {
            showCompletion();
        }
    }

    function startReportTimer() {
        timerOverlay.classList.remove('hidden');
        let timeLeft = 30;
        const totalTime = 30;
        
        const timerInterval = setInterval(() => {
            timeLeft--;
            timerCountdown.textContent = timeLeft;
            
            // Update circular progress
            const progress = ((totalTime - timeLeft) / totalTime) * 283;
            timerProgress.style.strokeDashoffset = 283 - progress;
            
            if (timeLeft <= 0) {
                clearInterval(timerInterval);
                timerOverlay.innerHTML = `
                    <div class="timer-content">
                        <h3><i class="fas fa-check-circle"></i> Report Ready</h3>
                        <p>You will now be redirected to the report.</p>
                        <a href="${reportUrl}" class="btn btn-primary">
                            <i class="fas fa-external-link-alt"></i>
                            View Report
                        </a>
                    </div>
                `;
                
                setTimeout(() => {
                    window.location.href = reportUrl;
                }, 3000);
            }
        }, 1000);
    }

    function showCompletion() {
        document.querySelector('.completion-message').classList.remove('hidden');
        document.querySelector('.report-section').classList.remove('hidden');
        
        const copyButton = document.getElementById('copyButton');
        copyButton.disabled = false;
        copyButton.querySelector('.tooltip').textContent = 'Click to copy report';

        startReportTimer();
    }

    // Initialize EventSource
    const eventSource = new EventSource('/stream');
    let lastProcessedStage = 0;

    eventSource.onmessage = function(event) {
        const log = event.data;
        document.getElementById('status').textContent = log;

        // Update stages based on log messages
        if (log.includes("Running server_emissions.py...") && lastProcessedStage < 1) {
            document.getElementById("step1").classList.add("completed");
            updateProgress(1);
            lastProcessedStage = 1;
        } else if (log.includes("Running GreenCodeRefiner.py...") && lastProcessedStage < 2) {
            document.getElementById("step2").classList.add("completed");
            updateProgress(2);
            lastProcessedStage = 2;
        } else if (log.includes("Running track_emissions.py...") && lastProcessedStage < 3) {
            document.getElementById("step3").classList.add("completed");
            updateProgress(3);
            lastProcessedStage = 3;
        }
    };

    eventSource.onerror = function(error) {
        console.error('EventSource failed:', error);
        eventSource.close();
    };

    // Copy functionality
    document.getElementById('copyButton').addEventListener('click', async () => {
        const reportText = document.getElementById('reportText').textContent;
        try {
            await navigator.clipboard.writeText(reportText);
            const button = document.getElementById('copyButton');
            button.classList.add('copied');
            button.innerHTML = `
                <i class="fas fa-check"></i>
                <span class="button-text">Copied!</span>
            `;
            setTimeout(() => {
                button.classList.remove('copied');
                button.innerHTML = `
                    <i class="fas fa-copy"></i>
                    <span class="button-text">Copy Report</span>
                `;
            }, 2000);
        } catch (err) {
            console.error('Failed to copy text:', err);
        }
    });
});