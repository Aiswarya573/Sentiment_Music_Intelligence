import tkinter as tk
from tkinter import font as tkfont
import threading

# Local imports (adjust path if running from project root)
try:
    from src.sentiment_model import predict_with_confidence
    from src.music_recommender import recommend_music
except ModuleNotFoundError:
    # Fallback when running gui.py directly from src/
    from src.sentiment_model import predict_with_confidence
    from src.music_recommender import recommend_music


# ─────────────────────────────────────────────
#  Color Palette
# ─────────────────────────────────────────────
BG        = "#0d0f1a"
CARD      = "#151827"
BORDER    = "#1e2235"
ACCENT    = "#7c6af7"        # purple
POS_CLR   = "#4ade80"        # green
NEG_CLR   = "#f87171"        # red
NEU_CLR   = "#facc15"        # yellow
TXT_PRI   = "#e8e9f0"
TXT_SEC   = "#6b7280"
ENTRY_BG  = "#1a1d2e"

SENTIMENT_COLORS = {
    "positive": POS_CLR,
    "negative": NEG_CLR,
    "neutral":  NEU_CLR,
}

SENTIMENT_EMOJI = {
    "positive": "😊",
    "negative": "😔",
    "neutral":  "😐",
}


# ─────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────

class MusicApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("🎵 Sentiment Music Intelligence")
        self.geometry("700x720")
        self.resizable(False, False)
        self.configure(bg=BG)

        # Fonts
        self.title_font   = tkfont.Font(family="Helvetica", size=22, weight="bold")
        self.sub_font     = tkfont.Font(family="Helvetica", size=11)
        self.label_font   = tkfont.Font(family="Helvetica", size=10, weight="bold")
        self.body_font    = tkfont.Font(family="Helvetica", size=10)
        self.mono_font    = tkfont.Font(family="Courier",   size=10)
        self.big_font     = tkfont.Font(family="Helvetica", size=40)

        self._build_ui()

    # ── UI Construction ──────────────────────────

    def _build_ui(self):
        # ── Header ──────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=30, pady=(28, 6))

        tk.Label(hdr, text="🎵 Mood Music", font=self.title_font,
                 bg=BG, fg=TXT_PRI).pack(anchor="w")
        tk.Label(hdr, text="Tell me how you feel — I'll find music that matches your vibe.",
                 font=self.sub_font, bg=BG, fg=TXT_SEC).pack(anchor="w", pady=(2, 0))

        self._divider()

        # ── Input Card ──────────────────────────
        inp_card = self._card()
        tk.Label(inp_card, text="How are you feeling today?",
                 font=self.label_font, bg=CARD, fg=TXT_SEC).pack(anchor="w", pady=(0, 6))

        entry_frame = tk.Frame(inp_card, bg=BORDER, bd=0)
        entry_frame.pack(fill="x")
        inner = tk.Frame(entry_frame, bg=ENTRY_BG)
        inner.pack(fill="x", padx=1, pady=1)

        self.entry = tk.Entry(inner, font=self.body_font, bg=ENTRY_BG,
                              fg=TXT_PRI, insertbackground=ACCENT,
                              relief="flat", bd=10)
        self.entry.pack(fill="x")
        self.entry.bind("<Return>", lambda e: self._analyze())

        self.btn = tk.Button(inp_card, text="Analyze Mood  →",
                             font=self.label_font, bg=ACCENT, fg="white",
                             relief="flat", cursor="hand2", activebackground="#6a58e0",
                             activeforeground="white", pady=10,
                             command=self._analyze)
        self.btn.pack(fill="x", pady=(12, 0))

        self._divider()

        # ── Sentiment Result Card ────────────────
        sent_card = self._card()
        sent_top  = tk.Frame(sent_card, bg=CARD)
        sent_top.pack(fill="x")

        self.emoji_lbl = tk.Label(sent_top, text="—", font=self.big_font,
                                  bg=CARD, fg=TXT_SEC)
        self.emoji_lbl.pack(side="left", padx=(0, 16))

        sent_info = tk.Frame(sent_top, bg=CARD)
        sent_info.pack(side="left", fill="both", expand=True)

        tk.Label(sent_info, text="Detected Sentiment",
                 font=self.label_font, bg=CARD, fg=TXT_SEC).pack(anchor="w")

        self.sent_lbl = tk.Label(sent_info, text="—",
                                 font=tkfont.Font(family="Helvetica", size=18, weight="bold"),
                                 bg=CARD, fg=TXT_SEC)
        self.sent_lbl.pack(anchor="w")

        self.conf_lbl = tk.Label(sent_info, text="Confidence: —",
                                 font=self.body_font, bg=CARD, fg=TXT_SEC)
        self.conf_lbl.pack(anchor="w", pady=(2, 0))

        # Probability bars
        bars_frame = tk.Frame(sent_card, bg=CARD)
        bars_frame.pack(fill="x", pady=(14, 0))
        self._prob_bars = {}
        for cls, color in [("positive", POS_CLR), ("negative", NEG_CLR), ("neutral", NEU_CLR)]:
            row = tk.Frame(bars_frame, bg=CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=cls.capitalize(), width=9, anchor="w",
                     font=self.body_font, bg=CARD, fg=TXT_SEC).pack(side="left")
            bar_bg = tk.Frame(row, bg=BORDER, height=8)
            bar_bg.pack(side="left", fill="x", expand=True)
            bar_fg = tk.Frame(bar_bg, bg=color, height=8, width=0)
            bar_fg.place(x=0, y=0, relheight=1)
            pct_lbl = tk.Label(row, text="0%", width=5, anchor="e",
                               font=self.mono_font, bg=CARD, fg=TXT_SEC)
            pct_lbl.pack(side="left", padx=(6, 0))
            self._prob_bars[cls] = (bar_bg, bar_fg, pct_lbl)

        self._divider()

        # ── Music Recommendations Card ───────────
        music_card = self._card()
        tk.Label(music_card, text="🎧 Recommended Music",
                 font=self.label_font, bg=CARD, fg=TXT_SEC).pack(anchor="w", pady=(0, 8))

        cols = tk.Frame(music_card, bg=CARD)
        cols.pack(fill="x")

        # Genres
        gen_frame = tk.Frame(cols, bg=CARD)
        gen_frame.pack(side="left", fill="both", expand=True)
        tk.Label(gen_frame, text="Genres", font=self.label_font,
                 bg=CARD, fg=ACCENT).pack(anchor="w")
        self.genre_frame = tk.Frame(gen_frame, bg=CARD)
        self.genre_frame.pack(anchor="w", pady=(6, 0))

        sep = tk.Frame(cols, bg=BORDER, width=1)
        sep.pack(side="left", fill="y", padx=16)

        # Songs
        song_frame = tk.Frame(cols, bg=CARD)
        song_frame.pack(side="left", fill="both", expand=True)
        tk.Label(song_frame, text="Songs", font=self.label_font,
                 bg=CARD, fg=ACCENT).pack(anchor="w")
        self.songs_frame = tk.Frame(song_frame, bg=CARD)
        self.songs_frame.pack(anchor="w", pady=(6, 0))

        # Status bar
        self.status_lbl = tk.Label(self, text="",
                                   font=self.body_font, bg=BG, fg=TXT_SEC)
        self.status_lbl.pack(pady=(4, 12))

    # ── Helpers ─────────────────────────────────

    def _divider(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20, pady=6)

    def _card(self) -> tk.Frame:
        frame = tk.Frame(self, bg=CARD, bd=0, relief="flat")
        frame.pack(fill="x", padx=20, pady=4)
        inner = tk.Frame(frame, bg=CARD)
        inner.pack(fill="x", padx=20, pady=14)
        return inner

    def _clear_frame(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def _set_status(self, msg, color=TXT_SEC):
        self.status_lbl.config(text=msg, fg=color)

    # ── Analysis ────────────────────────────────

    def _analyze(self):
        text = self.entry.get().strip()
        if not text:
            self._set_status("⚠️  Please enter how you're feeling.", NEU_CLR)
            return

        self.btn.config(state="disabled", text="Analyzing…")
        self._set_status("Running Naïve Bayes + Lexicon analysis…")
        threading.Thread(target=self._run_analysis, args=(text,), daemon=True).start()

    def _run_analysis(self, text):
        label, conf_pct, detail = predict_with_confidence(text)
        result = recommend_music(label)
        self.after(0, self._update_ui, label, conf_pct, detail, result)

    def _update_ui(self, label, conf_pct, detail, result):
        color = SENTIMENT_COLORS[label]
        emoji = SENTIMENT_EMOJI[label]

        # Sentiment section
        self.emoji_lbl.config(text=emoji, fg=color)
        self.sent_lbl.config(text=label.upper(), fg=color)
        self.conf_lbl.config(text=f"Confidence: {conf_pct}%  |  "
                                  f"NB: {detail['nb_label']}  |  "
                                  f"Lexicon: {detail['lexicon_label']} "
                                  f"(score {detail['lexicon_score']:+.1f})")

        # Probability bars
        bar_total = 340
        for cls, (bar_bg, bar_fg, pct_lbl) in self._prob_bars.items():
            pct = detail["nb_probs"].get(cls, 0)
            bar_bg.update_idletasks()
            width = int(bar_total * pct / 100)
            bar_fg.place(x=0, y=0, relheight=1, width=width)
            pct_lbl.config(text=f"{pct}%")

        # Genres
        self._clear_frame(self.genre_frame)
        for g in result["genre"]:
            tk.Label(self.genre_frame, text=f"● {g}",
                     font=self.body_font, bg=CARD, fg=color).pack(anchor="w", pady=1)

        # Songs
        self._clear_frame(self.songs_frame)
        for s in result["songs"]:
            tk.Label(self.songs_frame, text=f"♪  {s}",
                     font=self.body_font, bg=CARD, fg=TXT_PRI).pack(anchor="w", pady=1)

        self.btn.config(state="normal", text="Analyze Mood  →")
        self._set_status("✓ Analysis complete", POS_CLR)


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = MusicApp()
    app.mainloop()
