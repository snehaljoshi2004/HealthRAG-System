// MedQuery — main.js

document.addEventListener('DOMContentLoaded', function () {

    // ── Auto-hide flash alerts after 5s ──
    setTimeout(function () {
        document.querySelectorAll('.alert:not(.alert-permanent)').forEach(function (el) {
            el.style.transition = 'opacity 0.4s';
            el.style.opacity = '0';
            setTimeout(function () { if (el.parentNode) el.remove(); }, 400);
        });
    }, 5000);

    // ── Scroll-to-top button ──
    const scrollBtn = document.getElementById('scrollTopBtn');
    if (scrollBtn) {
        window.addEventListener('scroll', function () {
            scrollBtn.classList.toggle('visible', window.scrollY > 320);
        });
    }

    // ── Auto-resize textarea ──
    const textarea = document.getElementById('question');
    if (textarea) {
        textarea.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    }

    // ── Keyboard shortcut Ctrl+/ to focus search ──
    document.addEventListener('keydown', function (e) {
        if (e.ctrlKey && e.key === '/') {
            e.preventDefault();
            const q = document.getElementById('question');
            if (q) q.focus();
        }
    });

    // ── Stagger fade-up animation on result cards ──
    document.querySelectorAll('.fade-up').forEach(function (el, i) {
        el.style.animationDelay = (i * 0.07) + 's';
    });

    // ── Copy-to-clipboard (global) ──
    window.copyToClipboard = function (button, text) {
        navigator.clipboard.writeText(text).then(function () {
            const orig = button.innerHTML;
            button.innerHTML = '<i class="fas fa-check me-1"></i>Copied';
            button.classList.replace('btn-outline-secondary', 'btn-success');
            setTimeout(function () {
                button.innerHTML = orig;
                button.classList.replace('btn-success', 'btn-outline-secondary');
            }, 2000);
        }).catch(function () {
            // Fallback for older browsers
            const ta = document.createElement('textarea');
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
        });
    };

    // ── Bootstrap tooltips ──
    document.querySelectorAll('[title]').forEach(function (el) {
        new bootstrap.Tooltip(el);
    });
});