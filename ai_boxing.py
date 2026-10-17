"""
AI Boxing — Deep RL ile Boks Öğrenen İki Karakter
===================================================
pip install pygame numpy
python ai_boxing.py

İki karakter sıfırdan boks öğreniyor.
Her uzuv ayrı hasar alır, yorgunluk var, blok var, nakavt var.
PPO-lite ile gerçek gradient öğrenmesi.

KONTROLLER
  SPACE     : Duraklat
  LEFT/RIGHT: Hız (1x - 20x)
  R         : Sıfırla
  ESC       : Çıkış
"""

import pygame, sys, math, random, collections
import numpy as np

pygame.init()
W, H = 1280, 760
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("AI Boxing")
clock = pygame.time.Clock()

# ── Fontlar ────────────────────────────────────────────────────────────────────
FB  = pygame.font.SysFont("segoeui", 22, bold=True)
FM  = pygame.font.SysFont("segoeui", 15)
FS  = pygame.font.SysFont("segoeui", 13)
FXS = pygame.font.SysFont("segoeui", 11)

# ── Renkler ────────────────────────────────────────────────────────────────────
BG          = (15,  12,  25)
RING_FLOOR  = (60,  40,  20)
RING_EDGE   = (140, 100, 40)
ROPE_COL    = (200, 160, 60)
WHITE       = (255, 255, 255)
GOLD        = (255, 210, 50)
RED         = (220, 55,  55)
BLUE        = (55,  120, 220)
GREEN       = (50,  215, 110)
CYAN        = (60,  210, 210)
ORANGE      = (255, 150, 30)
GRAY        = (130, 140, 160)
DGRAY       = (35,  45,  70)
PANEL_BG    = (12,  18,  38)
PANEL_BD    = (45,  65,  120)
CROWD_COL   = (40,  25,  50)

# ── Layout ─────────────────────────────────────────────────────────────────────
RING_X  = 150
RING_Y  = 120
RING_W  = 760
RING_H  = 420
PANEL_X = RING_X + RING_W + 20
PANEL_W = W - PANEL_X - 10

# ── Simülasyon sabitleri ───────────────────────────────────────────────────────
MAX_HP        = 100.0
MAX_ENERGY    = 100.0
ROUND_TICKS   = 60 * 60      # 60 saniye round
PUNCH_RANGE   = 65
BLOCK_REDUCE  = 0.35          # blok enerji hasar azaltma

# ── Aksiyonlar ────────────────────────────────────────────────────────────────
# 0: Dur
# 1: İleri
# 2: Geri
# 3: Sol adım
# 4: Sağ adım
# 5: Sol yumruk (jab)
# 6: Sağ yumruk (cross)
# 7: Sol hook
# 8: Sağ hook
# 9: Uppercut
# 10: Blok
N_ACTIONS = 11

ACTION_NAMES = ["Dur","İleri","Geri","Sol adım","Sağ adım",
                "Jab","Cross","Sol hook","Sağ hook","Uppercut","Blok"]

# Yumruk parametreleri: (hasar, enerji_maliyeti, hız, uzuv)
PUNCH_PARAMS = {
    5:  (12, 6,  0.8, "jab"),
    6:  (18, 10, 0.6, "cross"),
    7:  (20, 12, 0.5, "left_hook"),
    8:  (20, 12, 0.5, "right_hook"),
    9:  (25, 15, 0.4, "uppercut"),
}

# ── Sinir Ağı (PPO-Lite) ──────────────────────────────────────────────────────
# Girdi: [kendi_x, kendi_y, kendi_hp, kendi_enerji, rakip_x, rakip_y,
#          rakip_hp, mesafe, açı, kendi_blokta_mı, rakip_blokta_mı,
#          son_hasar_aldım, yorgunluk] = 13 girdi
# Çıktı: N_ACTIONS olasılık (policy) + 1 değer (critic)

INP_SIZE = 13
HIDDEN   = 64

class PPONet:
    """Küçük ama gerçek PPO ağı: Actor + Critic birleşik"""
    def __init__(self):
        sc = 1.0/math.sqrt(INP_SIZE)
        # Paylaşımlı katman
        self.w1 = np.random.randn(INP_SIZE, HIDDEN)*sc
        self.b1 = np.zeros(HIDDEN)
        self.w2 = np.random.randn(HIDDEN, HIDDEN)*0.3
        self.b2 = np.zeros(HIDDEN)
        # Actor (policy) head
        self.wa = np.random.randn(HIDDEN, N_ACTIONS)*0.1
        self.ba = np.zeros(N_ACTIONS)
        # Critic (value) head
        self.wc = np.random.randn(HIDDEN, 1)*0.1
        self.bc = np.zeros(1)
        # Optimizer momentumu (Adam lite)
        self.lr = 0.003
        self.eps_clip = 0.2
        # Replay buffer
        self.buf_s  = []
        self.buf_a  = []
        self.buf_r  = []
        self.buf_lp = []   # log probabilities
        self.buf_v  = []   # value estimates

    def forward(self, x):
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        # Policy: softmax
        logits = h2 @ self.wa + self.ba
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / (exp_l.sum() + 1e-8)
        # Value
        value = float((h2 @ self.wc + self.bc)[0])
        return probs, value, h2

    def select_action(self, state, exploit=False):
        probs, value, _ = self.forward(state)
        if exploit:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(N_ACTIONS, p=probs))
        log_prob = math.log(probs[action] + 1e-8)
        return action, log_prob, value

    def push(self, s, a, r, lp, v):
        self.buf_s.append(s); self.buf_a.append(a)
        self.buf_r.append(r); self.buf_lp.append(lp)
        self.buf_v.append(v)
        if len(self.buf_s) > 2048:
            self.update()

    def update(self):
        if len(self.buf_s) < 32: return
        S  = np.array(self.buf_s,  dtype=np.float32)
        A  = np.array(self.buf_a,  dtype=np.int32)
        R  = np.array(self.buf_r,  dtype=np.float32)
        LP = np.array(self.buf_lp, dtype=np.float32)
        V  = np.array(self.buf_v,  dtype=np.float32)

        # Discounted returns
        gamma = 0.99
        returns = np.zeros_like(R)
        running = 0.0
        for i in reversed(range(len(R))):
            running = R[i] + gamma*running
            returns[i] = running
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = returns - V
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-8)

        # Mini-batch gradient updates
        n = len(S)
        for _ in range(4):
            idx = np.random.choice(n, min(128, n), replace=False)
            s=S[idx]; a=A[idx]; ret=returns[idx]; adv=advantages[idx]; old_lp=LP[idx]

            h1 = np.tanh(s @ self.w1 + self.b1)
            h2 = np.tanh(h1 @ self.w2 + self.b2)
            logits = h2 @ self.wa + self.ba
            logits -= logits.max(axis=1, keepdims=True)
            exp_l  = np.exp(logits)
            probs  = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-8)

            # New log probs for taken actions
            new_lp = np.log(probs[np.arange(len(a)), a] + 1e-8)
            ratio  = np.exp(new_lp - old_lp)

            # PPO clipped objective
            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1-self.eps_clip, 1+self.eps_clip)*adv
            actor_loss = -np.minimum(surr1, surr2)

            # Critic loss
            values = (h2 @ self.wc + self.bc).flatten()
            critic_loss = (ret - values)**2

            # Entropy bonus (keşif için)
            entropy = -(probs * np.log(probs + 1e-8)).sum(axis=1)

            total_loss = actor_loss + 0.5*critic_loss - 0.01*entropy

            # Backprop (basit gradient)
            lr = self.lr
            # Actor head
            d_probs = probs.copy()
            d_probs[np.arange(len(a)), a] -= 1.0
            d_probs *= (actor_loss[:,None] - 0.01*(1+np.log(probs+1e-8)))
            d_probs /= len(idx)

            dwa = h2.T @ d_probs; dba = d_probs.sum(0)
            self.wa -= lr * dwa; self.ba -= lr * dba

            # Critic head
            d_val = 2*(values - ret)[:,None]/len(idx)
            dwc = h2.T @ d_val; dbc = d_val.sum(0)
            self.wc -= lr * dwc; self.bc -= lr * dbc

            # Hidden layers (ortak)
            dh2 = (d_probs @ self.wa.T + d_val @ self.wc.T) * (1-h2**2)
            dw2 = h1.T @ dh2; db2 = dh2.sum(0)
            self.w2 -= lr*dw2; self.b2 -= lr*db2

            dh1 = (dh2 @ self.w2.T) * (1-h1**2)
            dw1 = s.T @ dh1; db1 = dh1.sum(0)
            self.w1 -= lr*dw1; self.b1 -= lr*db1

        self.buf_s.clear(); self.buf_a.clear(); self.buf_r.clear()
        self.buf_lp.clear(); self.buf_v.clear()

# ── Boksör ────────────────────────────────────────────────────────────────────
class Boxer:
    def __init__(self, side, color_body, color_glove, name):
        self.side       = side    # 0=sol, 1=sağ
        self.col_body   = color_body
        self.col_glove  = color_glove
        self.name       = name
        self.net        = PPONet()
        self.reset()

    def reset(self):
        if self.side == 0:
            self.x = RING_X + RING_W*0.28
        else:
            self.x = RING_X + RING_W*0.72
        self.y          = RING_Y + RING_H*0.55
        self.hp         = MAX_HP
        self.energy     = MAX_ENERGY
        self.blocking   = False
        self.punch_timer  = 0    # aktif yumruk süresi
        self.punch_type   = None
        self.punch_hit    = False
        self.stun_timer   = 0    # sersemlik
        self.knock_timer  = 0    # nakavt animasyonu
        self.knocked      = False
        self.total_hits   = 0
        self.total_damage = 0.0
        self.round_wins   = 0
        # Uzuv hasarları
        self.limb_hp = {"head":100,"body":100,"left_arm":100,"right_arm":100}
        # Öğrenme istatistikleri
        self.episode_reward = 0.0
        self.reward_history = []
        self.action_counts  = [0]*N_ACTIONS
        # Stil evrimi (zamanla hangi yumrukları tercih ediyor)
        self.style = "unknown"

    def get_state(self, opponent):
        # Normalize edilmiş girdi vektörü
        rx = (self.x - RING_X) / RING_W
        ry = (self.y - RING_Y) / RING_H
        ox = (opponent.x - RING_X) / RING_W
        oy = (opponent.y - RING_Y) / RING_H
        dx = opponent.x - self.x
        dy = opponent.y - self.y
        dist = math.hypot(dx, dy) / RING_W
        angle = math.atan2(dy, dx) / math.pi
        return np.array([
            rx, ry,
            self.hp / MAX_HP,
            self.energy / MAX_ENERGY,
            ox, oy,
            opponent.hp / MAX_HP,
            dist,
            angle,
            1.0 if self.blocking else 0.0,
            1.0 if opponent.blocking else 0.0,
            1.0 if self.stun_timer > 0 else 0.0,
            self.energy / MAX_ENERGY,   # yorgunluk tekrar
        ], dtype=np.float32)

    def act(self, state):
        if self.knocked or self.stun_timer > 0:
            return 0, 0.0, 0.0
        action, log_prob, value = self.net.select_action(state)
        self.action_counts[action] += 1
        return action, log_prob, value

    def apply_action(self, action, opponent):
        """Aksiyonu uygula, reward döndür"""
        reward = 0.0
        spd = 3.5

        if self.knocked: return reward
        if self.stun_timer > 0:
            self.stun_timer -= 1
            return reward

        self.blocking = False

        # Hareket
        if action == 1:   # İleri (rakibe doğru)
            dx = opponent.x - self.x
            dy = opponent.y - self.y
            d  = math.hypot(dx,dy)
            if d>1:
                self.x += dx/d*spd; self.y += dy/d*spd*0.3
            self.energy -= 0.3
        elif action == 2:  # Geri
            dx = opponent.x - self.x
            dy = opponent.y - self.y
            d  = math.hypot(dx,dy)
            if d>1:
                self.x -= dx/d*spd; self.y -= dy/d*spd*0.3
            self.energy -= 0.3
        elif action == 3:  # Sol adım
            self.x -= spd*0.8
            self.energy -= 0.2
        elif action == 4:  # Sağ adım
            self.x += spd*0.8
            self.energy -= 0.2
        elif action == 10:  # Blok
            self.blocking = True
            self.energy -= 0.15
            reward += 0.3

        # Ring sınırları
        margin = 30
        self.x = float(np.clip(self.x, RING_X+margin, RING_X+RING_W-margin))
        self.y = float(np.clip(self.y, RING_Y+margin, RING_Y+RING_H-margin))

        # Yumruk
        if action in PUNCH_PARAMS and self.punch_timer <= 0:
            base_dmg, energy_cost, speed, punch_name = PUNCH_PARAMS[action]
            if self.energy >= energy_cost:
                self.energy -= energy_cost
                self.punch_timer = int(20 / speed)
                self.punch_type  = action
                self.punch_hit   = False

                # Mesafe kontrolü
                dist = math.hypot(opponent.x-self.x, opponent.y-self.y)
                if dist < PUNCH_RANGE:
                    # İsabet!
                    dmg = base_dmg
                    # Uzuv hasarı
                    if action == 9:  # uppercut kafa
                        limb = "head"
                        dmg *= 1.3
                    elif action in (7,8):  # hook gövde
                        limb = "body"
                    else:
                        limb = "head" if random.random()>0.4 else "body"

                    # Blok azaltması
                    if opponent.blocking:
                        dmg *= BLOCK_REDUCE
                        limb = "left_arm" if self.side==0 else "right_arm"

                    opponent.limb_hp[limb] = max(0, opponent.limb_hp[limb]-dmg*0.5)
                    opponent.hp -= dmg
                    opponent.stun_timer = max(opponent.stun_timer, int(dmg/3))
                    self.punch_hit   = True
                    self.total_hits += 1
                    self.total_damage += dmg

                    reward += dmg * 0.15 + 2.0   # isabet ödülü
                    if opponent.hp <= 0:
                        reward += 50.0   # nakavt büyük ödül!
                else:
                    reward -= 0.5   # ıska ceza
            else:
                reward -= 0.1   # enerji yokken yumruk ceza

        # Enerji yenilenmesi
        self.energy = min(MAX_ENERGY, self.energy + 0.12)

        # Mesafeye göre küçük ödül/ceza
        dist = math.hypot(opponent.x-self.x, opponent.y-self.y)
        if dist < PUNCH_RANGE:
            reward += 0.1   # vuruş menzilinde olmak iyi
        elif dist > PUNCH_RANGE*3:
            reward -= 0.05  # çok uzak olmak kötü

        if self.punch_timer > 0: self.punch_timer -= 1

        return reward

    def check_knockdown(self):
        if self.hp <= 0 and not self.knocked:
            self.knocked    = True
            self.knock_timer = 180   # 3 saniye animasyon

    def tick_knockdown(self):
        if self.knock_timer > 0:
            self.knock_timer -= 1

    def compute_style(self):
        ac = self.action_counts
        total = sum(ac)+1
        jab_r  = ac[5]/total
        cross_r= ac[6]/total
        hook_r = (ac[7]+ac[8])/total
        up_r   = ac[9]/total
        blk_r  = ac[10]/total
        if blk_r > 0.25: self.style = "Defansif"
        elif up_r > 0.12: self.style = "Brawler"
        elif jab_r > 0.20: self.style = "Out-boxer"
        elif hook_r > 0.15: self.style = "Slugger"
        else: self.style = "All-around"


# ── Çizim: Boksör ──────────────────────────────────────────────────────────────
def draw_boxer(boxer, flip=False):
    x, y = int(boxer.x), int(boxer.y)
    bx = -1 if flip else 1    # yön (sol karakter sağa bakar)
    col = boxer.col_body
    glove = boxer.col_glove

    # Nakavt animasyonu
    if boxer.knocked:
        t = boxer.knock_timer / 180.0
        lean = int((1-t)*40)    # düşme açısı
        # Yatık çizim
        pygame.draw.ellipse(screen, col, (x-20, y+lean-5, 40, 20))
        pygame.draw.circle(screen, (220,185,150), (x + int(bx*5), y+lean-18), 12)
        # KO yazısı
        s = FB.render("KO!", True, RED)
        screen.blit(s, (x-s.get_width()//2, y+lean-55))
        return

    # Sersemlik efekti
    stun_offset = 0
    if boxer.stun_timer > 0:
        stun_offset = random.randint(-3,3)
        # Yıldızlar
        for i in range(3):
            sx = x + int(math.cos(pygame.time.get_ticks()*0.01+i*2.1)*20)
            sy = y - 55 + int(math.sin(pygame.time.get_ticks()*0.01+i*2.1)*8)
            pygame.draw.circle(screen, GOLD, (sx,sy), 4)

    x += stun_offset

    # Blok efekti
    if boxer.blocking:
        # Savunma pozisyonu parıltısı
        s = pygame.Surface((50,50), pygame.SRCALPHA)
        pygame.draw.circle(s, (100,150,255,40), (25,25), 25)
        screen.blit(s, (x-25, y-30))

    # Gövde
    pygame.draw.ellipse(screen, col, (x-14, y-38, 28, 42))

    # Baş
    head_col = (220,185,150)
    pygame.draw.circle(screen, head_col, (x, y-50), 14)

    # Yüz (dövüş ifadesi)
    pygame.draw.circle(screen, (30,20,10), (x+bx*4, y-52), 3)   # göz
    if boxer.stun_timer > 3:
        pygame.draw.line(screen,(180,80,80),(x-4,y-46),(x+4,y-46),2)  # ağız (eğri)
    else:
        pygame.draw.line(screen,(100,60,40),(x-4,y-45),(x+4,y-45),1)

    # Saç
    pygame.draw.arc(screen, (80,50,20), (x-14,y-65,28,22), 0, math.pi, 4)

    # Bacaklar
    leg_spread = 10
    pygame.draw.line(screen, col, (x-leg_spread,y+4), (x-leg_spread-3,y+28), 8)
    pygame.draw.line(screen, col, (x+leg_spread,y+4), (x+leg_spread+3,y+28), 8)
    # Ayakkabılar
    pygame.draw.ellipse(screen, (30,20,10),(x-leg_spread-8,y+22,18,10))
    pygame.draw.ellipse(screen, (30,20,10),(x+leg_spread-4, y+22,18,10))

    # Kollar
    punch_arm = boxer.punch_type if boxer.punch_timer>0 else None

    # Sol kol
    if punch_arm in (5,7) and not flip or punch_arm in (6,8) and flip:
        # Uzanmış yumruk
        ex = x + bx*50; ey = y - 25
        pygame.draw.line(screen, col, (x+bx*14,y-28),(ex,ey),8)
        pygame.draw.circle(screen, glove, (ex,ey), 10)
    else:
        # Normal duruş
        pygame.draw.line(screen, col, (x-12,y-28),(x-20+bx*5,y-12),8)
        pygame.draw.circle(screen, glove, (x-20+bx*5,y-10), 9)

    # Sağ kol
    if punch_arm in (6,8) and not flip or punch_arm in (5,7) and flip:
        ex = x + bx*52; ey = y - 28
        pygame.draw.line(screen, col, (x+bx*14,y-28),(ex,ey),8)
        pygame.draw.circle(screen, glove, (ex,ey), 10)
    elif punch_arm == 9:   # uppercut
        ex = x + bx*30; ey = y - 55
        pygame.draw.line(screen, col, (x+bx*10,y-15),(ex,ey),9)
        pygame.draw.circle(screen, glove, (ex,ey), 11)
    else:
        pygame.draw.line(screen, col, (x+12,y-28),(x+22-bx*5,y-10),8)
        pygame.draw.circle(screen, glove, (x+22-bx*5,y-8), 9)

    # Yumruk isabet efekti
    if boxer.punch_hit and boxer.punch_timer > 10:
        impact_x = x + bx*60
        impact_y = y - 30
        s2 = pygame.Surface((40,40), pygame.SRCALPHA)
        for r2 in range(5,20,4):
            pygame.draw.circle(s2,(255,220,50,80-r2*3),(20,20),r2,2)
        screen.blit(s2,(impact_x-20,impact_y-20))

    # HP barı üstünde
    bar_w = 50
    hp_ratio = max(0, boxer.hp/MAX_HP)
    bar_col = GREEN if hp_ratio>0.5 else (ORANGE if hp_ratio>0.25 else RED)
    pygame.draw.rect(screen,DGRAY,(x-bar_w//2,y-75,bar_w,7),border_radius=3)
    pygame.draw.rect(screen,bar_col,(x-bar_w//2,y-75,int(bar_w*hp_ratio),7),border_radius=3)

    # Enerji barı
    en_ratio = boxer.energy/MAX_ENERGY
    pygame.draw.rect(screen,DGRAY,(x-bar_w//2,y-66,bar_w,5),border_radius=2)
    pygame.draw.rect(screen,(60,120,255),(x-bar_w//2,y-66,int(bar_w*en_ratio),5),border_radius=2)

    # İsim etiketi
    ns = FXS.render(boxer.name, True, boxer.col_glove)
    screen.blit(ns,(x-ns.get_width()//2,y-88))


# ── Çizim: Ring ────────────────────────────────────────────────────────────────
def draw_ring():
    # Zemin
    pygame.draw.rect(screen,RING_FLOOR,(RING_X,RING_Y,RING_W,RING_H),border_radius=8)
    # Halı deseni
    for i in range(0,RING_W,40):
        c = (65,44,22) if (i//40)%2==0 else (58,38,18)
        pygame.draw.rect(screen,c,(RING_X+i,RING_Y,40,RING_H))

    # Merkez çizgi
    pygame.draw.line(screen,(80,55,25),(RING_X+RING_W//2,RING_Y),(RING_X+RING_W//2,RING_Y+RING_H),2)
    pygame.draw.circle(screen,(80,55,25),(RING_X+RING_W//2,RING_Y+RING_H//2),50,2)

    # Köşe direkleri
    post_r = 10
    for px,py in [(RING_X,RING_Y),(RING_X+RING_W,RING_Y),
                  (RING_X,RING_Y+RING_H),(RING_X+RING_W,RING_Y+RING_H)]:
        pygame.draw.circle(screen,RING_EDGE,(px,py),post_r)
        pygame.draw.circle(screen,(180,130,50),(px,py),post_r,2)

    # İpler
    rope_heights = [RING_Y+15, RING_Y+35, RING_Y+55]
    for ry in rope_heights:
        pygame.draw.line(screen,ROPE_COL,(RING_X,ry),(RING_X+RING_W,ry),3)
    rope_heights2 = [RING_Y+RING_H-15,RING_Y+RING_H-35,RING_Y+RING_H-55]
    for ry in rope_heights2:
        pygame.draw.line(screen,ROPE_COL,(RING_X,ry),(RING_X+RING_W,ry),3)

    # Kenar çerçeve
    pygame.draw.rect(screen,RING_EDGE,(RING_X,RING_Y,RING_W,RING_H),4,border_radius=8)


def draw_crowd():
    # Basit kalabalık silüeti
    for i in range(0,W,20):
        h2 = 30 + int(math.sin(i*0.3)*8) + int(math.sin(i*0.07)*15)
        col = (random.randint(35,55),random.randint(20,35),random.randint(45,65))
        pygame.draw.ellipse(screen,col,(i-8,RING_Y-h2-10,18,h2+10))
    # Alt kalabalık
    for i in range(0,W,22):
        h2 = 25 + int(math.sin(i*0.4)*6)
        pygame.draw.ellipse(screen,(40,25,50),(i-9,RING_Y+RING_H,20,h2))


def draw_lights():
    # Sahne ışıkları
    for lx in [RING_X+RING_W//4, RING_X+RING_W//2, RING_X+3*RING_W//4]:
        s = pygame.Surface((200,300),pygame.SRCALPHA)
        pygame.draw.polygon(s,(255,240,180,12),[(100,0),(20,299),(180,299)])
        screen.blit(s,(lx-100,RING_Y-100))


# ── Panel ──────────────────────────────────────────────────────────────────────
def draw_panel(b1, b2, round_num, tick, reward_hist1, reward_hist2):
    PX=PANEL_X; PW=PANEL_W
    pygame.draw.rect(screen,PANEL_BG,(PX,0,PW,H))
    pygame.draw.line(screen,PANEL_BD,(PX,0),(PX,H),2)

    y=8
    def dt(t,x,yy,col,f=FM,cx=False):
        s=f.render(str(t),True,col)
        if cx: x-=s.get_width()//2
        screen.blit(s,(int(x),int(yy)))
    def hline(yy): pygame.draw.line(screen,PANEL_BD,(PX+6,yy),(PX+PW-6,yy),1)

    dt("AI BOXING",PX+PW//2,y,GOLD,FB,cx=True); y+=26
    dt(f"Round {round_num}",PX+PW//2,y,WHITE,FS,cx=True); y+=18
    sec = (ROUND_TICKS-tick)//60
    dt(f"{sec//60:02d}:{sec%60:02d}",PX+PW//2,y,CYAN,FM,cx=True); y+=22
    hline(y); y+=8

    # Her boksör için bilgi
    for bx, flip, col in [(b1,False,b1.col_glove),(b2,True,b2.col_glove)]:
        dt(bx.name,PX+PW//2,y,col,FS,cx=True); y+=16

        # HP barı
        bw=PW-20; hp_r=max(0,bx.hp/MAX_HP)
        hcol=GREEN if hp_r>0.5 else (ORANGE if hp_r>0.25 else RED)
        pygame.draw.rect(screen,DGRAY,(PX+10,y,bw,12),border_radius=4)
        pygame.draw.rect(screen,hcol,(PX+10,y,int(bw*hp_r),12),border_radius=4)
        dt(f"HP {bx.hp:.0f}",PX+12,y+1,(10,10,10),FXS)
        y+=16

        # Enerji barı
        en_r=bx.energy/MAX_ENERGY
        pygame.draw.rect(screen,DGRAY,(PX+10,y,bw,8),border_radius=3)
        pygame.draw.rect(screen,(60,120,255),(PX+10,y,int(bw*en_r),8),border_radius=3)
        dt(f"ENJ {bx.energy:.0f}",PX+12,y,(10,10,10),FXS)
        y+=14

        # Uzuv hasarları (küçük barlar)
        for limb,lname in [("head","Kafa"),("body","Gövde"),("left_arm","Sol"),("right_arm","Sağ")]:
            lr=bx.limb_hp[limb]/100
            lc=(255,int(100+lr*100),int(50+lr*50))
            pygame.draw.rect(screen,DGRAY,(PX+10,y,bw//2-4,6),border_radius=2)
            pygame.draw.rect(screen,lc,(PX+10,y,int((bw//2-4)*lr),6),border_radius=2)
            dt(f"{lname}",PX+10,y+7,(130,130,150),FXS)
            y+=16

        # İstatistikler
        dt(f"İsabet: {bx.total_hits}  Hasar: {bx.total_damage:.0f}",PX+10,y,GRAY,FXS); y+=14
        dt(f"Stil: {bx.style}",PX+10,y,col,FXS); y+=14
        dt(f"Galibiyet: {bx.round_wins}",PX+10,y,GOLD,FXS); y+=18
        hline(y); y+=8

    # Ödül grafiği
    dt("Episode ödülleri:",PX+10,y,GRAY,FXS); y+=14
    gh=55; gw=PW-20
    pygame.draw.rect(screen,(22,32,55),(PX+10,y,gw,gh),border_radius=4)
    for hist,col2 in [(reward_hist1,b1.col_glove),(reward_hist2,b2.col_glove)]:
        if len(hist)>=2:
            mn=min(hist)-1; mx=max(hist)+1; rng=max(mx-mn,1)
            h2=hist[-gw:]
            pts=[(PX+10+int(ii*gw/max(len(h2)-1,1)),
                  y+gh-int((v-mn)/rng*(gh-4))-2) for ii,v in enumerate(h2)]
            if len(pts)>=2: pygame.draw.lines(screen,col2,False,pts,2)
    y+=gh+6

    # Aksiyon dağılımı (son boksör)
    hline(y); y+=6
    dt("Aksiyon dağılımı (Mavi):",PX+10,y,GRAY,FXS); y+=14
    total_ac = sum(b1.action_counts)+1
    for i,name in enumerate(ACTION_NAMES):
        ratio = b1.action_counts[i]/total_ac
        pygame.draw.rect(screen,DGRAY,(PX+10,y,gw,7),border_radius=2)
        pygame.draw.rect(screen,b1.col_glove,(PX+10,y,int(gw*ratio),7),border_radius=2)
        dt(f"{name} {ratio*100:.0f}%",PX+10,y+8,(120,130,150),FXS)
        y+=18
        if y>H-20: break


# ── Ana Döngü ─────────────────────────────────────────────────────────────────
def main():
    b1 = Boxer(0, (60,80,200),  (220,30,30),  "Mavi")
    b2 = Boxer(1, (200,70,40),  (30,180,30),  "Kirmizi")

    round_num   = 1
    tick        = 0
    sim_speed   = 3
    speed_opts  = [1,2,3,5,8,12,20]
    speed_idx   = 2
    paused      = False

    reward_hist1 = []
    reward_hist2 = []

    def end_round():
        nonlocal round_num
        # Kim kazandı?
        if b1.hp > b2.hp:
            b1.round_wins += 1
            winner = b1.name
        elif b2.hp > b1.hp:
            b2.round_wins += 1
            winner = b2.name
        else:
            winner = "Berabere"

        reward_hist1.append(b1.episode_reward)
        reward_hist2.append(b2.episode_reward)
        if len(reward_hist1)>100: reward_hist1.pop(0)
        if len(reward_hist2)>100: reward_hist2.pop(0)

        b1.compute_style(); b2.compute_style()
        b1.reset(); b2.reset()
        round_num += 1

    running=True
    while running:
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: running=False
            if ev.type==pygame.KEYDOWN:
                if ev.key==pygame.K_ESCAPE: running=False
                if ev.key==pygame.K_SPACE:  paused=not paused
                if ev.key==pygame.K_RIGHT:  speed_idx=min(speed_idx+1,len(speed_opts)-1); sim_speed=speed_opts[speed_idx]
                if ev.key==pygame.K_LEFT:   speed_idx=max(speed_idx-1,0); sim_speed=speed_opts[speed_idx]
                if ev.key==pygame.K_r:
                    b1.reset(); b2.reset(); round_num=1; tick=0
                    reward_hist1.clear(); reward_hist2.clear()

        if paused:
            screen.fill(BG); draw_crowd(); draw_lights(); draw_ring()
            draw_boxer(b1, flip=False); draw_boxer(b2, flip=True)
            draw_panel(b1,b2,round_num,tick,reward_hist1,reward_hist2)
            s=FB.render("DURAKLADI — SPACE devam",True,GOLD)
            screen.blit(s,(RING_X+RING_W//2-s.get_width()//2,RING_Y+RING_H//2-15))
            pygame.display.flip(); clock.tick(30); continue

        for _ in range(sim_speed):
            s1 = b1.get_state(b2)
            s2 = b2.get_state(b1)

            a1, lp1, v1 = b1.act(s1)
            a2, lp2, v2 = b2.act(s2)

            r1 = b1.apply_action(a1, b2)
            r2 = b2.apply_action(a2, b1)

            # Hasar aldıysa eksi ödül
            # (apply_action pozitif ödül verir vurana, ama vurulan da ceza alır)
            prev_hp1 = b1.hp; prev_hp2 = b2.hp
            # hp zaten apply_action içinde azaltıldı
            dmg_to_1 = max(0, prev_hp1 - b1.hp)
            dmg_to_2 = max(0, prev_hp2 - b2.hp)
            r1 -= dmg_to_1 * 0.1
            r2 -= dmg_to_2 * 0.1

            b1.episode_reward += r1
            b2.episode_reward += r2

            b1.net.push(s1,a1,r1,lp1,v1)
            b2.net.push(s2,a2,r2,lp2,v2)

            b1.check_knockdown(); b2.check_knockdown()
            b1.tick_knockdown(); b2.tick_knockdown()

            tick+=1

            # Round bitiş koşulları
            if b1.knocked or b2.knocked or tick>=ROUND_TICKS:
                end_round(); tick=0

        # ── Çizim ──────────────────────────────────────────────────────────
        screen.fill(BG)
        draw_crowd()
        draw_lights()
        draw_ring()
        draw_boxer(b1, flip=False)
        draw_boxer(b2, flip=True)
        draw_panel(b1,b2,round_num,tick,reward_hist1,reward_hist2)

        # Hız göstergesi
        s=FS.render(f"Hız: {sim_speed}x  Round: {round_num}",True,GRAY)
        screen.blit(s,(8,H-18))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit(); sys.exit()

if __name__=="__main__":
    main()
