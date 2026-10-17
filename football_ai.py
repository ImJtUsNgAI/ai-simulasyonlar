"""
AI Futbol Simülasyonu - Reinforcement Learning
Gereksinim: pip install pygame numpy
Çalıştır: python football_ai.py
"""

import pygame
import numpy as np
import math
import sys
import random

pygame.init()

# ── Ekran ──────────────────────────────────────────────────────────────────────
W, H = 1920, 1080
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("AI Futbol - RL Eğitimi")
clock = pygame.time.Clock()
font_big   = pygame.font.SysFont("segoeui", 26, bold=True)
font_med   = pygame.font.SysFont("segoeui", 18)
font_small = pygame.font.SysFont("segoeui", 14)

# ── Renkler ────────────────────────────────────────────────────────────────────
C = {
    "bg":         (15,  25,  45),
    "grass":      (34,  100, 34),
    "grass2":     (30,  90,  30),
    "line":       (255, 255, 255),
    "goal_post":  (255, 255, 255),
    "goal_net":   (200, 220, 255),
    "agent_top":  (130, 180, 255),
    "agent_face": (80,  140, 240),
    "agent_side": (50,  100, 200),
    "agent_eye":  (255, 255, 255),
    "agent_pupil":(20,  20,  60),
    "ball":       (240, 240, 240),
    "ball_patch": (30,  30,  30),
    "shadow":     (0,   0,   0),
    "panel_bg":   (20,  30,  55),
    "panel_brd":  (60,  80,  130),
    "reward_pos": (50,  220, 120),
    "reward_neg": (220, 60,  60),
    "gold":       (255, 210, 50),
    "white":      (255, 255, 255),
    "red":        (220, 60,  60),
    "green":      (50,  220, 120),
    "cyan":       (80,  220, 220),
    "orange":     (255, 160, 40),
    "gray":       (140, 150, 170),
    "timer_ok":   (80,  200, 100),
    "timer_warn": (255, 180, 40),
    "timer_danger":(220, 60, 60),
}

# ── Saha Ayarları ──────────────────────────────────────────────────────────────
FIELD_W   = 1000   # simülasyon koordinatları
FIELD_H   = 800

# İzometrik projeksiyon merkezi
ISO_CX = 560
ISO_CY = 390
ISO_SX = 0.70   # x ölçeği
ISO_SY = 0.38   # y (derinlik) ölçeği
ISO_Z  = 0.55   # z (yükseklik) ölçeği

# Kale (sağ taraf)
GOAL_X  = FIELD_W - 10
GOAL_Y1 = FIELD_H // 2 - 60
GOAL_Y2 = FIELD_H // 2 + 60
GOAL_D  = 50    # derinlik (içeri)
GOAL_H_3D = 55  # yükseklik

# ── İzometrik Projeksiyon ──────────────────────────────────────────────────────
def iso(fx, fy, fz=0):
    sx = (fx - fy) * ISO_SX + ISO_CX
    sy = (fx + fy) * ISO_SY - fz * ISO_Z + ISO_CY
    return (int(sx), int(sy))

# ── Sinir Ağı (Beyin) ──────────────────────────────────────────────────────────
class Brain:
    """
    Küçük ileri beslemeli sinir ağı.
    Girdi (7) → Gizli (12) → Çıktı (4)
    Çıktılar: [vx_delta, vy_delta, kick_x, kick_y]
    """
    def __init__(self):
        self.w1 = np.random.randn(7, 12) * 0.5
        self.b1 = np.zeros(12)
        self.w2 = np.random.randn(12, 4) * 0.5
        self.b2 = np.zeros(4)

    def forward(self, inp):
        h = np.tanh(inp @ self.w1 + self.b1)
        return np.tanh(h @ self.w2 + self.b2)

    def clone(self):
        b = Brain()
        b.w1 = self.w1.copy(); b.b1 = self.b1.copy()
        b.w2 = self.w2.copy(); b.b2 = self.b2.copy()
        return b

    def mutate(self, rate=0.3, strength=0.4):
        for arr in [self.w1, self.b1, self.w2, self.b2]:
            mask = np.random.rand(*arr.shape) < rate
            arr += mask * np.random.randn(*arr.shape) * strength

# ── Top ────────────────────────────────────────────────────────────────────────
class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = FIELD_W * 0.35 + random.uniform(-30, 30)
        self.y = FIELD_H * 0.5  + random.uniform(-40, 40)
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, 0.5)
        self.r  = 12
        self.angle = 0.0   # görsel dönme

    def step(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vx *= 0.97
        self.vy *= 0.97
        self.angle += self.vx * 0.25

        # Saha sınırları
        if self.x < self.r:
            self.x = self.r;  self.vx *= -0.65
        if self.x > FIELD_W - self.r:
            self.x = FIELD_W - self.r; self.vx *= -0.65
        if self.y < self.r:
            self.y = self.r;  self.vy *= -0.65
        if self.y > FIELD_H - self.r:
            self.y = FIELD_H - self.r; self.vy *= -0.65

    def is_goal(self):
        return (self.x >= GOAL_X - 5 and
                GOAL_Y1 <= self.y <= GOAL_Y2)

    def dist_to_goal(self):
        cx = GOAL_X
        cy = (GOAL_Y1 + GOAL_Y2) / 2
        return math.hypot(self.x - cx, self.y - cy)

# ── Ajan ───────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, brain=None):
        self.brain = brain or Brain()
        self.reset()

    def reset(self):
        self.x  = 80 + random.uniform(-20, 20)
        self.y  = FIELD_H / 2 + random.uniform(-50, 50)
        self.vx = 0.0; self.vy = 0.0
        self.score       = 0.0
        self.touch_timer = 0       # topa dokunmadan geçen tick
        self.age         = 0
        self.goals       = 0
        self.bob         = random.uniform(0, math.pi * 2)
        self.flash       = 0
        self.flash_color = C["white"]
        self.last_ball_dist_to_goal = 9999.0
        self.squish      = 1.0
        self.reward_log  = []      # son ödül olayları

    # ── Bir Adım ──────────────────────────────────────────────────────────────
    def step(self, ball: Ball, dt=1.0):
        self.age         += dt
        self.touch_timer += dt
        self.bob         += 0.10

        # Girdi vektörü (normalize)
        dx_b  = (ball.x - self.x)  / FIELD_W
        dy_b  = (ball.y - self.y)  / FIELD_H
        bvx   = ball.vx / 8.0
        bvy   = ball.vy / 8.0
        gx    = (GOAL_X - ball.x) / FIELD_W   # topun kaleye uzaklığı
        gy    = ((GOAL_Y1+GOAL_Y2)/2 - ball.y) / FIELD_H
        self_vx = self.vx / 7.0

        inp = np.array([dx_b, dy_b, bvx, bvy, gx, gy, self_vx], dtype=np.float32)
        out = self.brain.forward(inp)          # [-1,1] x4

        # Hareket
        ax = out[0] * 3.5
        ay = out[1] * 3.5
        self.vx = np.clip(self.vx * 0.80 + ax, -7, 7)
        self.vy = np.clip(self.vy * 0.80 + ay, -7, 7)
        self.x  = np.clip(self.x + self.vx, 8, FIELD_W - 8)
        self.y  = np.clip(self.y + self.vy, 8, FIELD_H - 8)

        speed = math.hypot(self.vx, self.vy)
        self.squish = 1.0 + speed * 0.012

        # ── Topa dokun ────────────────────────────────────────────────────────
        dist_to_ball = math.hypot(self.x - ball.x, self.y - ball.y)

        # Topa yaklaştıkça sürekli küçük ödül
        proximity_reward = max(0, (150 - dist_to_ball) / 150) * 0.05
        self.score += proximity_reward

        if dist_to_ball < 26:
            self.touch_timer = 0
            nx = (ball.x - self.x) / max(dist_to_ball, 1)
            ny = (ball.y - self.y) / max(dist_to_ball, 1)
            kick_str = 5.0 + abs(out[2]) * 2.0
            ball.vx = nx * kick_str + out[2] * 1.5
            ball.vy = ny * kick_str + out[3] * 1.5
            self.score += 8
            self.flash = 10; self.flash_color = C["gold"]
            self._add_log("+8 Dokunma!", C["gold"])

        # ── Topu kaleye yaklaştırma ödülü ─────────────────────────────────────
        ball_goal_dist = ball.dist_to_goal()
        improvement = self.last_ball_dist_to_goal - ball_goal_dist
        if improvement > 0:
            r_val = improvement * 0.15
            self.score += r_val
            if improvement > 3:
                self._add_log(f"+{r_val:.1f} Kaleye yaklaştı!", C["green"])
        self.last_ball_dist_to_goal = ball_goal_dist

        # ── 15 saniye topa dokunmazsa ceza ────────────────────────────────────
        TOUCH_LIMIT = 15 * 60
        if self.touch_timer >= TOUCH_LIMIT:
            self.score -= 40
            self.touch_timer = 0
            self.flash = 14; self.flash_color = C["red"]
            self._add_log("-40 Zaman aşımı!", C["red"])

        if self.flash > 0:
            self.flash -= 1

    def _add_log(self, msg, color):
        self.reward_log.append((msg, color, 120))  # 120 tick göster

    def tick_logs(self):
        self.reward_log = [(m,c,t-1) for m,c,t in self.reward_log if t > 1]

# ── Evrimsel Seçilim ──────────────────────────────────────────────────────────
def evolve(agent: Agent) -> Agent:
    new_brain = agent.brain.clone()
    gen = agent.goals
    rate     = max(0.08, 0.4 - gen * 0.02)
    strength = max(0.1,  0.5 - gen * 0.02)
    new_brain.mutate(rate, strength)
    return Agent(new_brain)

# ── Çizim: 3D Saha ────────────────────────────────────────────────────────────
def draw_field():
    # Zemin poligonu
    corners = [iso(0,0), iso(FIELD_W,0), iso(FIELD_W,FIELD_H), iso(0,FIELD_H)]
    pygame.draw.polygon(screen, C["grass"], corners)

    # Şerit çizgiler
    stripes = 8
    for i in range(stripes):
        x0 = FIELD_W / stripes * i
        x1 = FIELD_W / stripes * (i + 1)
        col = C["grass"] if i % 2 == 0 else C["grass2"]
        pts = [iso(x0,0), iso(x1,0), iso(x1,FIELD_H), iso(x0,FIELD_H)]
        pygame.draw.polygon(screen, col, pts)

    # Saha çizgileri
    def field_line(ax,ay,bx,by,w=2):
        pygame.draw.line(screen, (*C["line"],120), iso(ax,ay), iso(bx,by), w)

    field_line(0,0, FIELD_W,0)
    field_line(0,FIELD_H, FIELD_W,FIELD_H)
    field_line(0,0, 0,FIELD_H)
    field_line(FIELD_W,0, FIELD_W,FIELD_H)
    field_line(FIELD_W//2, 0, FIELD_W//2, FIELD_H)
    # Orta daire
    steps = 48
    circle_pts = []
    for i in range(steps):
        a = 2*math.pi*i/steps
        cx = FIELD_W/2 + math.cos(a)*60
        cy = FIELD_H/2 + math.sin(a)*60
        circle_pts.append(iso(cx,cy))
    pygame.draw.lines(screen, (*C["line"],80), True, circle_pts, 2)

def draw_goal_3d():
    """Kaleyi 3D kutup + file olarak çiz."""
    gx = GOAL_X
    y1 = GOAL_Y1; y2 = GOAL_Y2
    d  = GOAL_D
    zh = GOAL_H_3D

    # Zemin dikdörtgeni (file zemini)
    net_floor = [iso(gx,y1), iso(gx,y2), iso(gx+d,y2), iso(gx+d,y1)]
    pygame.draw.polygon(screen, (180,200,230, 60), net_floor)

    # Direkler (4 köşe)
    posts = [
        [(gx,y1,0),(gx,y1,zh)],
        [(gx,y2,0),(gx,y2,zh)],
        [(gx+d,y1,0),(gx+d,y1,zh)],
        [(gx+d,y2,0),(gx+d,y2,zh)],
    ]
    for (ax,ay,az),(bx,by,bz) in posts:
        pygame.draw.line(screen, C["goal_post"], iso(ax,ay,az), iso(bx,by,bz), 4)

    # Üst yatay kirişler
    bars = [
        [(gx,y1,zh),(gx,y2,zh)],           # ön üst
        [(gx+d,y1,zh),(gx+d,y2,zh)],       # arka üst
        [(gx,y1,zh),(gx+d,y1,zh)],          # yan üst
        [(gx,y2,zh),(gx+d,y2,zh)],          # yan üst
        [(gx,y1,0),(gx+d,y1,0)],            # yan alt
        [(gx,y2,0),(gx+d,y2,0)],            # yan alt
    ]
    for (ax,ay,az),(bx,by,bz) in bars:
        pygame.draw.line(screen, C["goal_post"], iso(ax,ay,az), iso(bx,by,bz), 3)

    # File (grid)
    steps_y = 6; steps_z = 4
    for i in range(steps_y+1):
        t = i/steps_y
        fy = y1 + (y2-y1)*t
        pygame.draw.line(screen, (*C["goal_net"],80),
                         iso(gx,fy,0), iso(gx,fy,zh), 1)
        pygame.draw.line(screen, (*C["goal_net"],50),
                         iso(gx,fy,0), iso(gx+d,fy,0), 1)
        pygame.draw.line(screen, (*C["goal_net"],50),
                         iso(gx,fy,zh), iso(gx+d,fy,zh), 1)
    for j in range(steps_z+1):
        t = j/steps_z
        fz = zh*t
        pygame.draw.line(screen, (*C["goal_net"],80),
                         iso(gx,y1,fz), iso(gx,y2,fz), 1)
        pygame.draw.line(screen, (*C["goal_net"],50),
                         iso(gx,y1,fz), iso(gx+d,y1,fz), 1)
        pygame.draw.line(screen, (*C["goal_net"],50),
                         iso(gx,y2,fz), iso(gx+d,y2,fz), 1)

# ── Çizim: Ajan (sevimli 3D küp) ─────────────────────────────────────────────
def draw_agent(agent: Agent):
    ax = agent.x; ay = agent.y
    sz = 18 * agent.squish
    zh = 28
    bob_z = math.sin(agent.bob) * 3

    col_top  = agent.flash_color if agent.flash > 0 else C["agent_top"]
    col_face = agent.flash_color if agent.flash > 0 else C["agent_face"]
    col_side = agent.flash_color if agent.flash > 0 else C["agent_side"]

    # Gölge
    shadow_pts = [
        iso(ax-sz*0.8, ay-sz*0.4),
        iso(ax+sz*0.8, ay-sz*0.4),
        iso(ax+sz*0.8, ay+sz*0.4),
        iso(ax-sz*0.8, ay+sz*0.4),
    ]
    s = pygame.Surface((W,H), pygame.SRCALPHA)
    pygame.draw.ellipse(s, (0,0,0,60),
        pygame.Rect(iso(ax,ay)[0]-int(sz*1.2),
                    iso(ax,ay)[1]-int(sz*0.3),
                    int(sz*2.4), int(sz*0.7)))
    screen.blit(s,(0,0))

    # Küpün 8 köşesi (3D)
    offsets = [(-1,-1),( 1,-1),( 1, 1),(-1, 1)]
    bot = [iso(ax+o[0]*sz, ay+o[1]*sz*0.5, bob_z)       for o in offsets]
    top = [iso(ax+o[0]*sz, ay+o[1]*sz*0.5, zh+bob_z)    for o in offsets]

    # Sol yüz
    pygame.draw.polygon(screen, col_side, [bot[0],bot[3],top[3],top[0]])
    # Ön yüz
    pygame.draw.polygon(screen, col_face, [bot[1],bot[2],top[2],top[1]])
    # Üst yüz
    pygame.draw.polygon(screen, col_top, [top[0],top[1],top[2],top[3]])

    # Kenarlıklar
    for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)]:
        all_pts = bot+top
        pygame.draw.line(screen, (0,0,0,100), all_pts[a], all_pts[b], 1)

    # Göz
    ex, ey = iso(ax+sz*0.5, ay-sz*0.3, zh*0.75+bob_z)
    pygame.draw.circle(screen, C["agent_eye"],   (ex,ey), 5)
    pygame.draw.circle(screen, C["agent_pupil"], (ex+1,ey), 2)

    # Gülümseme
    smile_rect = pygame.Rect(ex-5, ey+4, 10, 6)
    pygame.draw.arc(screen, C["agent_eye"], smile_rect, math.pi, 2*math.pi, 2)

# ── Çizim: Top ────────────────────────────────────────────────────────────────
def draw_ball(ball: Ball):
    bx,by = iso(ball.x, ball.y, ball.r)
    r = 10

    # Gölge
    sx,sy = iso(ball.x, ball.y, 0)
    s = pygame.Surface((W,H), pygame.SRCALPHA)
    pygame.draw.ellipse(s, (0,0,0,70), (sx-r, sy-r//3, r*2, r//1.5))
    screen.blit(s,(0,0))

    pygame.draw.circle(screen, C["ball"], (bx,by), r)
    for i in range(5):
        a = ball.angle + i*math.pi*2/5
        px = bx + int(math.cos(a)*r*0.5)
        py = by + int(math.sin(a)*r*0.5)
        pygame.draw.circle(screen, C["ball_patch"], (px,py), int(r*0.28))
    pygame.draw.circle(screen, (80,80,80), (bx,by), r, 2)

# ── Çizim: Sağ Panel (İstatistikler + Ödül Grafiği) ──────────────────────────
def draw_panel(agent: Agent, generation: int, total_goals: int,
               best_score: float, score_history: list, touch_pct: float):
    PX = W - 260; PY = 10; PW = 250; PH = H - 20

    # Panel arkaplanı
    s = pygame.Surface((PW, PH), pygame.SRCALPHA)
    s.fill((20,30,55,210))
    screen.blit(s,(PX,PY))
    pygame.draw.rect(screen, C["panel_brd"], (PX,PY,PW,PH), 2, border_radius=10)

    def txt(text, x, y, color, f=font_med):
        screen.blit(f.render(text, True, color), (x,y))

    y = PY + 12
    txt("AI FUTBOL SIM", PX+10, y, C["gold"], font_big); y+=34

    # Çizgi
    pygame.draw.line(screen, C["panel_brd"], (PX+8,y), (PX+PW-8,y), 1); y+=8

    # Bilgiler
    items = [
        ("Nesil",        str(generation),          C["cyan"]),
        ("Toplam Gol",   str(total_goals),          C["gold"]),
        ("Bu ep. skor",  f"{agent.score:.0f}",     C["white"]),
        ("En iyi skor",  f"{best_score:.0f}",      C["green"]),
        ("AI Yaşı",      f"{agent.age//60:.0f}s",  C["gray"]),
    ]
    for label, val, col in items:
        txt(label+":", PX+10, y, C["gray"], font_small)
        txt(val, PX+150, y, col, font_med)
        y += 24

    y += 6
    pygame.draw.line(screen, C["panel_brd"], (PX+8,y), (PX+PW-8,y), 1); y+=10

    # Topa Dokunma Zamanlayıcısı
    txt("Topa dokunma süresi:", PX+10, y, C["gray"], font_small); y+=18
    touch_sec = agent.touch_timer / 60
    bar_w = int(PW - 20)
    ratio = min(touch_sec / 15.0, 1.0)
    bar_col = C["timer_ok"] if ratio < 0.5 else (C["timer_warn"] if ratio < 0.85 else C["timer_danger"])
    pygame.draw.rect(screen, (40,50,80), (PX+10, y, bar_w, 16), border_radius=4)
    if ratio > 0:
        pygame.draw.rect(screen, bar_col, (PX+10, y, int(bar_w*ratio), 16), border_radius=4)
    txt(f"{touch_sec:.1f}s / 15s", PX+10, y+18, bar_col, font_small); y+=38

    # Kaleye Uzaklık Çubuğu
    pygame.draw.line(screen, C["panel_brd"], (PX+8,y), (PX+PW-8,y), 1); y+=8
    txt("Topun kaleye uzaklığı:", PX+10, y, C["gray"], font_small); y+=18
    from math import hypot
    ball_dist = hypot(agent.last_ball_dist_to_goal, 0) if hasattr(agent,'last_ball_dist_to_goal') else 999
    max_d = math.hypot(FIELD_W, FIELD_H)
    dist_ratio = 1.0 - min(agent.last_ball_dist_to_goal / max_d, 1.0)
    pygame.draw.rect(screen, (40,50,80), (PX+10, y, bar_w, 16), border_radius=4)
    d_col = C["green"] if dist_ratio > 0.6 else (C["orange"] if dist_ratio > 0.3 else C["red"])
    if dist_ratio > 0:
        pygame.draw.rect(screen, d_col, (PX+10, y, int(bar_w*dist_ratio), 16), border_radius=4)
    txt(f"{agent.last_ball_dist_to_goal:.0f} birim", PX+10, y+18, d_col, font_small); y+=38

    # Ödül Geçmişi Grafiği
    pygame.draw.line(screen, C["panel_brd"], (PX+8,y), (PX+PW-8,y), 1); y+=8
    txt("Skor geçmisi (nesiller):", PX+10, y, C["gray"], font_small); y+=18
    gh = 100; gw = PW - 20
    pygame.draw.rect(screen, (30,40,70), (PX+10, y, gw, gh), border_radius=4)

    if len(score_history) >= 2:
        mn = min(score_history); mx = max(score_history)
        rng = max(mx - mn, 1)
        pts = []
        for i, sc in enumerate(score_history[-gw:]):
            px2 = PX + 10 + int(i * gw / max(len(score_history[-gw:])-1,1))
            py2 = y + gh - int((sc - mn) / rng * (gh-4)) - 2
            pts.append((px2, py2))
        if len(pts) >= 2:
            pygame.draw.lines(screen, C["green"], False, pts, 2)

    # Skor referans çizgisi
    pygame.draw.line(screen, (*C["panel_brd"],120), (PX+10,y+gh//2),(PX+10+gw,y+gh//2),1)
    y += gh + 10

    # Son ödül olayları
    pygame.draw.line(screen, C["panel_brd"], (PX+8,y), (PX+PW-8,y), 1); y+=8
    txt("Son olaylar:", PX+10, y, C["gray"], font_small); y+=16
    for msg, col, _ in agent.reward_log[-4:]:
        txt(msg, PX+10, y, col, font_small); y+=16

# ── Çizim: Gol Animasyonu ─────────────────────────────────────────────────────
def draw_goal_flash(timer):
    if timer <= 0: return
    alpha = int(min(timer * 8, 180))
    s = pygame.Surface((W,H), pygame.SRCALPHA)
    s.fill((255,220,30, alpha//3))
    screen.blit(s,(0,0))
    scale = 1.0 + (60-timer)*0.02
    text = font_big.render("GOL!", True, C["gold"])
    tw, th = text.get_size()
    scaled = pygame.transform.scale(text, (int(tw*scale), int(th*scale)))
    screen.blit(scaled, (W//2 - scaled.get_width()//2, H//2 - scaled.get_height()//2 - 40))

# ── Ana Döngü ─────────────────────────────────────────────────────────────────
def main():
    agent         = Agent()
    ball          = Ball()
    generation    = 1
    total_goals   = 0
    best_score    = 0.0
    score_history = []
    episode_max   = 18 * 60   # 18 saniye episode
    episode_tick  = 0
    goal_flash    = 0
    sim_speed     = 3
    paused        = False

    # Hız seçenekleri
    speed_options = [1, 2, 3, 5, 8, 10, 15]
    speed_idx     = 2

    log_msgs = [("Eğitim başladı! AI futbol öğreniyor...", C["cyan"], 300)]

    def add_log(msg, col):
        log_msgs.append((msg, col, 240))

    running = True
    while running:
        # ── Olaylar ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_RIGHT:
                    speed_idx = min(speed_idx+1, len(speed_options)-1)
                    sim_speed = speed_options[speed_idx]
                if event.key == pygame.K_LEFT:
                    speed_idx = max(speed_idx-1, 0)
                    sim_speed = speed_options[speed_idx]
                if event.key == pygame.K_r:
                    agent = Agent(); ball = Ball()
                    generation=1; total_goals=0; best_score=0
                    score_history.clear()
                    add_log("Sıfırlandı!", C["orange"])
                if event.key == pygame.K_m:
                    agent.brain.mutate(0.7, 0.6)
                    add_log("Zorla mutasyon!", C["cyan"])

        if paused:
            screen.fill(C["bg"])
            txt = font_big.render("DURAKLADI  [SPACE] devam", True, C["gold"])
            screen.blit(txt, (W//2-txt.get_width()//2, H//2))
            pygame.display.flip(); clock.tick(30)
            continue

        # ── Simülasyon adımları ────────────────────────────────────────────────
        for _ in range(sim_speed):
            agent.step(ball)
            ball.step()
            agent.tick_logs()
            episode_tick += 1

            # Gol?
            if ball.is_goal():
                total_goals += 1
                agent.score += 500
                agent.goals += 1
                goal_flash = 60
                add_log(f"GOL! Toplam: {total_goals}", C["gold"])
                ball.reset()
                agent.touch_timer = 0

            # Episode bitti?
            if episode_tick >= episode_max:
                if agent.score > best_score:
                    best_score = agent.score
                score_history.append(agent.score)
                add_log(f"Nesil {generation} → skor: {agent.score:.0f}", C["gray"])
                agent = evolve(agent)
                ball.reset()
                generation += 1
                episode_tick = 0

            if goal_flash > 0: goal_flash -= 1

        log_msgs = [(m,c,t-sim_speed) for m,c,t in log_msgs if t > 0]

        # ── Çizim ─────────────────────────────────────────────────────────────
        screen.fill(C["bg"])

        draw_field()
        draw_goal_3d()
        draw_ball(ball)
        draw_agent(agent)
        draw_goal_flash(goal_flash)

        draw_panel(agent, generation, total_goals, best_score,
                   score_history, agent.touch_timer/60)

        # Alt bilgi çubuğu
        info_y = H - 34
        pygame.draw.rect(screen, (15,22,40), (0, info_y-4, W-260, 38))
        controls = [
            ("[SOL/SAG]", f"Hız: {sim_speed}x"),
            ("[SPACE]", "Durdur"),
            ("[R]", "Sıfırla"),
            ("[M]", "Mutasyon"),
            ("[ESC]", "Çıkış"),
        ]
        cx = 12
        for key, desc in controls:
            k = font_small.render(key, True, C["cyan"])
            d = font_small.render(" "+desc+"  ", True, C["gray"])
            screen.blit(k, (cx, info_y))
            cx += k.get_width()
            screen.blit(d, (cx, info_y))
            cx += d.get_width()

        # Sol üst log
        for i,(msg,col,_) in enumerate(log_msgs[-5:]):
            screen.blit(font_small.render(msg, True, col), (12, 12+i*16))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
