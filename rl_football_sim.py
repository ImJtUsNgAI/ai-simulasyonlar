"""
AI Football Simulator v4
========================
Requirements : pip install pygame numpy
Run          : python football_ai.py

CONTROLS
  LEFT / RIGHT  : Simulation speed
  UP / DOWN     : Add / remove agent
  1 - 5         : Switch level (brain is transferred)
  A             : Algorithm menu
  S             : Settings / Reward editor
  P             : Toggle player control (WASD + Space to kick)
  O             : Objectives panel
  SPACE         : Pause
  R             : Reset (fresh start)
  M             : Force mutation
  ESC           : Quit
"""

import pygame, sys, math, random, collections, copy
import numpy as np

try:
    import config as CFG
    LEVELS     = CFG.LEVELS
    ALGORITHMS = CFG.ALGORITHMS
    FW = CFG.FIELD_W; FH = CFG.FIELD_H
    W  = CFG.WINDOW_W; H  = CFG.WINDOW_H
    OBJECTIVES = CFG.OBJECTIVES
except ImportError:
    # fallback defaults if config.py missing
    FW=800; FH=560; W=1400; H=860
    OBJECTIVES = {}
    ALGORITHMS = {
        "neuroevolution":{"name":"Neuroevolution","short":"NE","desc":"Best brain is parent."},
        "tournament":{"name":"Tournament","short":"TS","desc":"3 candidates, best wins."},
        "elitism":{"name":"Elitism","short":"EL","desc":"Top 2 copied directly."},
        "mentor":{"name":"Mentor","short":"MT","desc":"Best blended into others."},
        "annealing":{"name":"Annealing","short":"SA","desc":"High mutation early, fades."},
        "ppo_lite":{"name":"PPO-Lite","short":"PPO","desc":"Gradient from replay buffer."},
        "dqn_lite":{"name":"DQN-Lite","short":"DQN","desc":"Q-learning, 9 actions."},
        "marl":{"name":"MARL","short":"MARL","desc":"Team score shared."},
        "perfection":{"name":"Perfection Mode","short":"PERF","desc":"Only the absolute best survives. Zero tolerance for failure.","color":(255,80,80)},
        "boost":     {"name":"Boost Mode",       "short":"BST", "desc":"Ball magnet + 3x episode length. Learns faster by design.","color":(255,200,0)},
        "cheat":     {"name":"Cheat Mode",       "short":"CHT", "desc":"AI gets exact ball direction as extra input. Near-perfect info.","color":(255,120,0)},
        "god":       {"name":"God Mode",         "short":"GOD", "desc":"AI nudged toward correct action every step. Learns in minutes.","color":(255,50,50)},
    }
    LEVELS = {
        1:{"name":"Level 1","desc":"Touch the ball","n_blue":1,"n_red":0,"episode_s":20,"touch_limit":15,
           "has_goalkeeper":False,"match_mode":False,
           "reward":{"touch":20,"no_touch_tick":-0.10,"ball_to_goal":0.0,"goal":0,"timeout":-50,"proximity":0.018,"dribble_max":3.0,
                   "approach_bonus":0.012,"goal_approach_mult":1.0,"consecutive_touch":5.0,"wall_penalty":-0.5,
                   "idle_penalty":-0.02,"speed_bonus":0.008,"center_field_bonus":0.005,"pass_bonus":8.0,
                   "near_goal_bonus":0.10,"shot_on_target":15.0}},
        2:{"name":"Level 2","desc":"Shoot on goal","n_blue":1,"n_red":0,"episode_s":25,"touch_limit":15,
           "has_goalkeeper":False,"match_mode":False,
           "reward":{"touch":10,"no_touch_tick":-0.06,"ball_to_goal":0.20,"goal":400,"timeout":-40,"proximity":0.018,"dribble_max":3.0,
                   "approach_bonus":0.012,"goal_approach_mult":1.0,"consecutive_touch":5.0,"wall_penalty":-0.5,
                   "idle_penalty":-0.02,"speed_bonus":0.008,"center_field_bonus":0.005,"pass_bonus":8.0,
                   "near_goal_bonus":0.10,"shot_on_target":15.0}},
        3:{"name":"Level 3","desc":"Goalkeeper present","n_blue":1,"n_red":0,"episode_s":30,"touch_limit":15,
           "has_goalkeeper":True,"match_mode":False,
           "reward":{"touch":8,"no_touch_tick":-0.05,"ball_to_goal":0.22,"goal":600,"timeout":-40,"proximity":0.018,"dribble_max":3.0,
                   "approach_bonus":0.012,"goal_approach_mult":1.0,"consecutive_touch":5.0,"wall_penalty":-0.5,
                   "idle_penalty":-0.02,"speed_bonus":0.008,"center_field_bonus":0.005,"pass_bonus":8.0,
                   "near_goal_bonus":0.10,"shot_on_target":15.0}},
        4:{"name":"Level 4","desc":"2v1 passing","n_blue":2,"n_red":0,"episode_s":35,"touch_limit":15,
           "has_goalkeeper":True,"match_mode":False,
           "reward":{"touch":6,"no_touch_tick":-0.04,"ball_to_goal":0.24,"goal":700,"timeout":-40,"proximity":0.018,"dribble_max":3.0,
                   "approach_bonus":0.012,"goal_approach_mult":1.0,"consecutive_touch":5.0,"wall_penalty":-0.5,
                   "idle_penalty":-0.02,"speed_bonus":0.008,"center_field_bonus":0.005,"pass_bonus":8.0,
                   "near_goal_bonus":0.10,"shot_on_target":15.0}},
        5:{"name":"Level 5","desc":"Full 2v2 match","n_blue":2,"n_red":2,"episode_s":90,"touch_limit":20,
           "has_goalkeeper":False,"match_mode":True,
           "reward":{"touch":5,"no_touch_tick":-0.03,"ball_to_goal":0.26,"goal":1000,"timeout":-35,"proximity":0.018,"dribble_max":3.0,
                   "approach_bonus":0.012,"goal_approach_mult":1.0,"consecutive_touch":5.0,"wall_penalty":-0.5,
                   "idle_penalty":-0.02,"speed_bonus":0.008,"center_field_bonus":0.005,"pass_bonus":8.0,
                   "near_goal_bonus":0.10,"shot_on_target":15.0}},
    }

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("RL Football Simulator")
clock = pygame.time.Clock()

FB  = pygame.font.SysFont("segoeui", 21, bold=True)
FM  = pygame.font.SysFont("segoeui", 15)
FS  = pygame.font.SysFont("segoeui", 13)
FXS = pygame.font.SysFont("segoeui", 11)

# ── Colors ────────────────────────────────────────────────────────────────────
BG        = (12,  20,  40)
FIELD_COL = (30,  95,  30)
FIELD_COL2= (26,  84,  26)
WHITE     = (255, 255, 255)
GOLD      = (255, 210,  50)
RED       = (215,  50,  50)
GREEN     = ( 45, 210, 105)
CYAN      = ( 60, 210, 210)
ORANGE    = (255, 150,  30)
GRAY      = (135, 145, 165)
DGRAY     = ( 38,  48,  72)
PBG       = ( 16,  24,  48)   # panel background
PBD       = ( 50,  70, 125)   # panel border
BLUE_LT   = (115, 170, 255)
BLUE_MD   = ( 65, 125, 230)
BLUE_DK   = ( 38,  85, 190)
RED_LT    = (255, 125, 125)
RED_MD    = (205,  55,  55)
RED_DK    = (155,  28,  28)
PURPLE    = (155,  85, 215)
YELLOW    = (235, 215,  45)
TEAL      = ( 35, 195, 175)
PINK      = (250,  95, 175)
PLAYER_C  = (255, 230,  80)   # human player color

# ── Geometry ──────────────────────────────────────────────────────────────────
GX  = FW-8;  GY1  = FH//2-70; GY2  = FH//2+70
LGX = 8;     LGY1 = FH//2-70; LGY2 = FH//2+70
GD  = 55;    GH3  = 65

# ── Layout: field area vs right panel ─────────────────────────────────────────
PANEL_W   = 300          # right panel width
FIELD_X   = 0            # field area starts at x=0
FIELD_AREA_W = W - PANEL_W
# isometric origin sits in field area
ICX = FIELD_AREA_W // 2 + 60
ICY = H // 2 + 30
ISX, ISY, ISZ = 0.68, 0.36, 0.52

def iso(fx, fy, fz=0):
    sx = (fx - fy)*ISX + ICX
    sy = (fx + fy)*ISY - fz*ISZ + ICY
    return (int(sx), int(sy))

ALGO_KEYS = list(ALGORITHMS.keys())

# ── Neural network ────────────────────────────────────────────────────────────
class Brain:
    def __init__(self, inp=7):
        self.inp = inp
        sc = 1.0 / math.sqrt(inp)
        self.w1 = np.random.randn(inp, 16)*sc; self.b1 = np.zeros(16)
        self.w2 = np.random.randn(16, 4)*0.3;  self.b2 = np.zeros(4)
    def forward(self, x):
        return np.tanh(np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2)
    def clone(self):
        b = Brain(self.inp)
        b.w1=self.w1.copy(); b.b1=self.b1.copy()
        b.w2=self.w2.copy(); b.b2=self.b2.copy()
        return b
    def mutate(self, rate=0.25, strength=0.35):
        for arr in [self.w1, self.b1, self.w2, self.b2]:
            arr += (np.random.rand(*arr.shape)<rate)*np.random.randn(*arr.shape)*strength
    def blend(self, other, alpha=0.5):
        self.w1=alpha*self.w1+(1-alpha)*other.w1; self.b1=alpha*self.b1+(1-alpha)*other.b1
        self.w2=alpha*self.w2+(1-alpha)*other.w2; self.b2=alpha*self.b2+(1-alpha)*other.b2

class PPOBuffer:
    def __init__(self): self.s=[]; self.a=[]; self.r=[]
    def push(self, s, a, r):
        self.s.append(s); self.a.append(a); self.r.append(r)
        if len(self.s) > 2000: self.s.pop(0); self.a.pop(0); self.r.pop(0)
    def clear(self): self.s=[]; self.a=[]; self.r=[]
    def update(self, brain, lr=0.003, epochs=3):
        if len(self.s) < 16: return
        S=np.array(self.s,dtype=np.float32); A=np.array(self.a,dtype=np.float32)
        R=np.array(self.r,dtype=np.float32); R=(R-R.mean())/(R.std()+1e-8)
        for _ in range(epochs):
            idx=np.random.choice(len(S),min(64,len(S)),replace=False)
            s=S[idx]; a=A[idx]; r=R[idx]
            h=np.tanh(s@brain.w1+brain.b1); out=np.tanh(h@brain.w2+brain.b2)
            adv=r[:,None]*(a-out); dout=adv*(1-out**2)
            brain.w2+=lr*(h.T@dout)/len(idx); brain.b2+=lr*dout.mean(0)
            dh=(dout@brain.w2.T)*(1-h**2)
            brain.w1+=lr*(s.T@dh)/len(idx); brain.b1+=lr*dh.mean(0)

ACTS_DQN = [(1,0),(-1,0),(0,1),(0,-1),(0.7,0.7),(-0.7,0.7),(0.7,-0.7),(-0.7,-0.7),(0,0)]

class DQNBrain:
    def __init__(self, inp=7):
        self.inp=inp; sc=1.0/math.sqrt(inp)
        self.w1=np.random.randn(inp,24)*sc; self.b1=np.zeros(24)
        self.w2=np.random.randn(24,9)*0.3;  self.b2=np.zeros(9)
        self.eps=1.0; self.buf=collections.deque(maxlen=3000)
    def qv(self, x): return np.tanh(x@self.w1+self.b1)@self.w2+self.b2
    def act(self, x): return random.randint(0,8) if random.random()<self.eps else int(np.argmax(self.qv(x)))
    def push(self, s,a,r,s2): self.buf.append((s,a,r,s2))
    def update(self, lr=0.005, gamma=0.95, bs=32):
        if len(self.buf)<bs: return
        d=random.sample(self.buf,bs)
        S=np.array([x[0] for x in d],dtype=np.float32); A=np.array([x[1] for x in d],dtype=np.int32)
        R=np.array([x[2] for x in d],dtype=np.float32); S2=np.array([x[3] for x in d],dtype=np.float32)
        Q=self.qv(S); Q2=self.qv(S2); T=Q.copy()
        for i in range(bs): T[i,A[i]]=R[i]+gamma*Q2[i].max()
        h=np.tanh(S@self.w1+self.b1); e=T-Q
        self.w2+=lr*(h.T@e)/bs; self.b2+=lr*e.mean(0)
        dh=(e@self.w2.T)*(1-h**2); self.w1+=lr*(S.T@dh)/bs; self.b1+=lr*dh.mean(0)
    def decay(self): self.eps=max(0.05,self.eps*0.999)
    def clone(self):
        b=DQNBrain(self.inp); b.w1=self.w1.copy(); b.b1=self.b1.copy()
        b.w2=self.w2.copy(); b.b2=self.b2.copy(); b.eps=max(0.05,self.eps*0.98); return b

# ── Ball ──────────────────────────────────────────────────────────────────────
class Ball:
    R = 12
    def __init__(self): self.last_toucher=None; self.reset()
    def reset(self):
        self.x=FW*0.35+random.uniform(-20,20); self.y=FH*0.5+random.uniform(-35,35)
        self.vx=random.uniform(-0.3,0.3); self.vy=random.uniform(-0.3,0.3)
        self.angle=0.0; self.last_toucher=None
    def step(self):
        self.x+=self.vx; self.y+=self.vy; self.vx*=0.970; self.vy*=0.970
        self.angle+=self.vx*0.20; r=self.R
        # Sol duvar — kale bölgesinde sektirme, geç
        if self.x<r:
            if LGY1<=self.y<=LGY2:
                pass  # sol kaleye giriyor, sektirme
            else:
                self.x=r; self.vx*=-0.62
        # Sağ duvar — kale bölgesinde sektirme, geç
        if self.x>FW-r:
            if GY1<=self.y<=GY2:
                pass  # sağ kaleye giriyor, sektirme
            else:
                self.x=FW-r; self.vx*=-0.62
        if self.y<r:    self.y=r;    self.vy*=-0.62
        if self.y>FH-r: self.y=FH-r; self.vy*=-0.62
    def in_right(self): return self.x>=FW-self.R and GY1<=self.y<=GY2
    def in_left(self):  return self.x<=self.R    and LGY1<=self.y<=LGY2
    def dr(self): return math.hypot(self.x-GX,  self.y-(GY1+GY2)/2)
    def dl(self): return math.hypot(self.x-LGX, self.y-(LGY1+LGY2)/2)

# ── Human Player ──────────────────────────────────────────────────────────────
class HumanPlayer:
    SZ=18; H3=28
    def __init__(self):
        self.x=FW*0.5; self.y=FH*0.5
        self.vx=0.0; self.vy=0.0
        self.bob=0.0; self.squish=1.0
        self.score=0.0; self.goals=0
        self.flash=0; self.fc=PLAYER_C
        self.team=2   # neutral team (doesn't affect AI rewards)
        self.imitation_data=[]   # stores (state, action) pairs for imitation learning

    def update(self, ball, keys):
        self.bob+=0.12
        spd=5.5
        ax=ay=0.0
        if keys[pygame.K_w]: ay=-spd
        if keys[pygame.K_s]: ay= spd
        if keys[pygame.K_a]: ax=-spd
        if keys[pygame.K_d]: ax= spd
        self.vx=ax*0.85; self.vy=ay*0.85
        self.x=float(np.clip(self.x+self.vx, 8, FW-8))
        self.y=float(np.clip(self.y+self.vy, 8, FH-8))
        self.squish=1.0+math.hypot(self.vx,self.vy)*0.010
        if self.flash>0: self.flash-=1

        # Kick with SPACE
        d=math.hypot(self.x-ball.x, self.y-ball.y)
        if keys[pygame.K_SPACE] and d<28:
            nx=(ball.x-self.x)/max(d,1); ny=(ball.y-self.y)/max(d,1)
            ball.vx=nx*7.0; ball.vy=ny*7.0
            ball.last_toucher=2
            self.flash=10; self.fc=PLAYER_C
            # Store for imitation learning
            state=[
                (ball.x-self.x)/FW,(ball.y-self.y)/FH,
                ball.vx/8,ball.vy/8,
                (GX-ball.x)/FW,((GY1+GY2)/2-ball.y)/FH,self.vx/7
            ]
            action=[nx,ny,0.0,0.0]
            self.imitation_data.append((np.array(state,dtype=np.float32),
                                        np.array(action,dtype=np.float32)))
            if len(self.imitation_data)>500: self.imitation_data.pop(0)

    def reset(self):
        self.x=FW*0.5; self.y=FH*0.5; self.vx=0; self.vy=0
        self.score=0.0; self.goals=0

# ── AI Agent ──────────────────────────────────────────────────────────────────
class Agent:
    SZ=17; H3=26
    def __init__(self, brain, team=0, sx=90, sy=None,
                 ct=BLUE_LT, cf=BLUE_MD, cs=BLUE_DK):
        self.brain=brain; self.team=team; self.sx=sx; self.sy=sy or FH//2
        self.ct=ct; self.cf=cf; self.cs=cs; self.touch_limit=15
        self.pbuf=PPOBuffer(); self.pi=None; self.pa=None; self.reset()
    def reset(self):
        mg=60
        if self.team==0: self.x=float(np.clip(self.sx+random.uniform(-15,15),mg,FW//2-mg))
        else:            self.x=float(np.clip(self.sx+random.uniform(-15,15),FW//2+mg,FW-mg))
        self.y=float(np.clip(self.sy+random.uniform(-40,40),mg,FH-mg))
        self.vx=0.0; self.vy=0.0; self.score=0.0; self.touch_timer=0
        self.tracking=False; self.prev_dist=None; self.dd=None
        self.age=0; self.goals=0; self.bob=random.uniform(0,math.pi*2)
        self.flash=0; self.fc=WHITE; self.squish=1.0; self.ev=[]
        self.pi=None; self.pa=None
    def ae(self, msg, col): self.ev.append([msg,col,90])
    def te(self):
        for e in self.ev: e[2]-=1
        self.ev=[e for e in self.ev if e[2]>0]
    def get_state(self, ball, algo='neuroevolution', gk=None, mates=None, opps=None):
        if self.team==0: gdx=(GX-ball.x)/FW;  gdy=((GY1+GY2)/2-ball.y)/FH
        else:            gdx=(LGX-ball.x)/FW;  gdy=((LGY1+LGY2)/2-ball.y)/FH
        v=[(ball.x-self.x)/FW,(ball.y-self.y)/FH,ball.vx/8,ball.vy/8,gdx,gdy,self.vx/7]
        v+=[(gk.x-self.x)/FW,(gk.y-self.y)/FH] if gk else [0.0,0.0]
        v+=[(mates[0].x-self.x)/FW,(mates[0].y-self.y)/FH] if mates else [0.0,0.0]
        v+=[(opps[0].x-self.x)/FW,(opps[0].y-self.y)/FH] if opps else [0.0,0.0]
        # CHEAT: topun kaleye tam yönünü normalize et ve ekle
        if algo=="cheat":
            goal_cx = GX if self.team==0 else LGX
            goal_cy = (GY1+GY2)/2
            gd = math.hypot(goal_cx-ball.x, goal_cy-ball.y)
            v[4] = (goal_cx-ball.x)/max(gd,1)   # normalize yön x (override)
            v[5] = (goal_cy-ball.y)/max(gd,1)   # normalize yön y (override)
        return np.array(v[:self.brain.inp],dtype=np.float32)
    def step(self, ball, rcfg, algo, gk=None, mates=None, opps=None):
        self.age+=1; self.touch_timer+=1; self.bob+=0.11
        if self.flash>0: self.flash-=1
        xa=self.get_state(ball,algo,gk,mates,opps)
        if algo=="dqn_lite" and isinstance(self.brain,DQNBrain):
            ai=self.brain.act(xa); ax,ay=ACTS_DQN[ai]; out=np.array([ax,ay,0.0,0.0])
        else:
            out=self.brain.forward(xa); ai=None
        db=math.hypot(self.x-ball.x,self.y-ball.y)

        # BOOST: güçlü top mıknatısı
        pull_range = 400 if algo=="boost" else 200
        pull_str   = 2.5 if algo=="boost" else 0.8
        pull=max(0,(pull_range-db)/pull_range)*pull_str
        px=(ball.x-self.x)/max(db,1)*pull; py=(ball.y-self.y)/max(db,1)*pull

        # GOD: her adımda doğru yöne hafif it
        if algo=="god":
            # Topa doğru git, yoksa kaleye
            goal_cx = GX if self.team==0 else LGX
            goal_cy = (GY1+GY2)/2
            if db > 30:
                # topa git
                correct_vx = (ball.x-self.x)/max(db,1)*3.0
                correct_vy = (ball.y-self.y)/max(db,1)*3.0
            else:
                # topu kaleye at
                gd = math.hypot(goal_cx-ball.x, goal_cy-ball.y)
                correct_vx = (goal_cx-ball.x)/max(gd,1)*3.0
                correct_vy = (goal_cy-ball.y)/max(gd,1)*3.0
            # AI çıktısını doğru yönle harmanlal (0.3 AI, 0.7 god)
            out = np.array([
                out[0]*0.3 + correct_vx*0.7/3.8,
                out[1]*0.3 + correct_vy*0.7/3.8,
                out[2], out[3]
            ])

        self.vx=float(np.clip(self.vx*0.80+out[0]*3.8+px,-7.5,7.5))
        self.vy=float(np.clip(self.vy*0.80+out[1]*3.8+py,-7.5,7.5))
        self.x=float(np.clip(self.x+self.vx,8,FW-8))
        self.y=float(np.clip(self.y+self.vy,8,FH-8))
        self.squish=1.0+math.hypot(self.vx,self.vy)*0.011
        prx=rcfg.get("proximity",0.018); dmx=rcfg.get("dribble_max",3.0)
        apb=rcfg.get("approach_bonus",0.012); gam=rcfg.get("goal_approach_mult",1.0)
        wpen=rcfg.get("wall_penalty",-0.5); ipen=rcfg.get("idle_penalty",-0.02)
        spdb=rcfg.get("speed_bonus",0.008); cfb=rcfg.get("center_field_bonus",0.005)
        ngb=rcfg.get("near_goal_bonus",0.10); sot=rcfg.get("shot_on_target",15.0)

        # ── REWARD: only positive signals, no per-tick penalties ──────────
        sr = 0.0

        spd_self = math.hypot(self.vx, self.vy)

        # 1) Proximity: closer to ball = small reward
        prox = max(0, (150 - db) / 150) * prx
        self.score += prox; sr += prox

        # 2) Approach: moving toward ball = reward
        if not hasattr(self,'_prev_db'): self._prev_db = db
        approach = self._prev_db - db
        if approach > 0:
            ab = approach * apb; self.score += ab; sr += ab
        self._prev_db = db

        touched=db<25
        if touched:
            self.touch_timer=0
            nx=(ball.x-self.x)/max(db,1); ny=(ball.y-self.y)/max(db,1)
            ks=5.5+abs(float(out[2]))*2.5
            ball.vx=nx*ks+float(out[2])*1.8; ball.vy=ny*ks+float(out[3])*1.8
            ball.last_toucher=self.team
            if not hasattr(ball,'_prev_toucher'): ball._prev_toucher=None
            ball._prev_toucher=ball.last_toucher
            bspd=math.hypot(ball.vx,ball.vy)
            drib=min(bspd*0.4,dmx) if bspd>1.5 else 0.0

            # Consecutive touch bonus
            ctb=rcfg.get("consecutive_touch",5.0)
            if hasattr(self,'_consec'): self._consec+=1
            else: self._consec=1
            consec_bns=ctb if self._consec>=2 else 0.0

            # Shot on target: ball heading toward goal after kick
            goal_cx=GX if self.team==0 else LGX
            goal_cy=(GY1+GY2)/2
            dx_g=goal_cx-ball.x; dy_g=goal_cy-ball.y
            dot=ball.vx*dx_g+ball.vy*dy_g
            sot_bns=sot if dot>0 and bspd>3 else 0.0

            tr=rcfg["touch"]+drib+consec_bns+sot_bns
            self.score+=tr; sr+=tr
            self.prev_dist=ball.dr() if self.team==0 else ball.dl()
            self.tracking=True; self.flash=12; self.fc=GOLD
            self.ae(f"+{rcfg['touch']} Touch!",GOLD)
            if sot_bns>0: self.ae(f"+{sot_bns:.0f} Shot on target!",CYAN)
        else:
            self._consec=0

        if self.tracking and not touched and rcfg["ball_to_goal"]>0:
            cur=ball.dr() if self.team==0 else ball.dl()
            if self.prev_dist is not None:
                imp=self.prev_dist-cur
                if imp>0:
                    bns=imp*rcfg["ball_to_goal"]*gam; self.score+=bns; sr+=bns
                    if imp>5: self.ae(f"+{bns:.1f} To goal!",GREEN)
                # Near goal bonus: ball is close to goal
                if cur<120: ng=ngb*(1-cur/120); self.score+=ng; sr+=ng
            self.prev_dist=cur; self.dd=cur
            # Pass bonus: if teammate touched ball before and we continue
            pb=rcfg.get("pass_bonus",8.0)
            if hasattr(ball,'_prev_toucher') and ball._prev_toucher!=self.team and ball._prev_toucher is not None:
                self.score+=pb; sr+=pb; ball._prev_toucher=None
            if math.hypot(ball.vx,ball.vy)<0.22: self.tracking=False
        if self.touch_timer>=self.touch_limit*60:
            pen = -15.0   # small fixed penalty, not from config
            self.score+=pen; sr+=pen
            self.touch_timer=0; self.tracking=False; self.flash=16; self.fc=RED
            self.ae("-15 Timeout!",RED)
        if algo=="ppo_lite": self.pbuf.push(xa.copy(),out.copy(),sr)
        if algo=="dqn_lite" and isinstance(self.brain,DQNBrain):
            if self.pi is not None: self.brain.push(self.pi,self.pa,sr,xa); self.brain.update()
            self.pi=xa.copy(); self.pa=ai; self.brain.decay()

# ── Goalkeeper ────────────────────────────────────────────────────────────────
class Goalkeeper:
    SZ=15; H3=24; SPD=3.8
    def __init__(self):
        self.home_x=GX-22; self.ct=YELLOW; self.cf=(210,180,20); self.cs=(170,140,10)
        self.team=1; self.reset()
    def reset(self):
        self.x=float(self.home_x); self.y=float((GY1+GY2)/2)
        self.vx=0.0; self.vy=0.0; self.bob=random.uniform(0,math.pi*2)
        self.flash=0; self.fc=WHITE; self.squish=1.0; self.ev=[]
    def step(self, ball):
        self.bob+=0.09; ty=float(np.clip(ball.y,GY1+5,GY2-5))
        self.y+=float(np.clip(ty-self.y,-self.SPD,self.SPD))
        tx=self.home_x+(ball.x-self.home_x)*0.06
        self.x+=(tx-self.x)*0.09; self.x=float(np.clip(self.x,self.home_x-18,self.home_x+12))
    def block(self, ball):
        d=math.hypot(self.x-ball.x,self.y-ball.y)
        if d<23:
            nx=(ball.x-self.x)/max(d,1); ny=(ball.y-self.y)/max(d,1)
            spd=math.hypot(ball.vx,ball.vy)
            ball.vx=nx*max(spd,3.5); ball.vy=ny*max(spd,3.5)*0.32; return True
        return False

# ── Evolution ─────────────────────────────────────────────────────────────────
def _ch(tmpl, brain):
    c=Agent(brain,tmpl.team,tmpl.sx,tmpl.sy,tmpl.ct,tmpl.cf,tmpl.cs)
    c.touch_limit=tmpl.touch_limit; return c

def evolve_agents(agents, gen, algo):
    if not agents: return agents
    sc=sorted(agents,key=lambda a:a.score,reverse=True); n=len(agents)
    def mk(parent, r, s):
        nb=parent.clone(); nb.mutate(r,s); return nb
    # Mutation rate stays meaningful — never shrinks to near-zero
    r = 0.25
    s = 0.30
    if algo=="neuroevolution":
        return [_ch(ag, mk(sc[0].brain,r,s)) for ag in agents]
    elif algo=="tournament":
        r2=max(0.06,0.38-gen*0.006); s2=max(0.07,0.36-gen*0.005)
        new=[]
        for ag in agents:
            p=max(random.sample(agents,min(3,n)),key=lambda a:a.score)
            new.append(_ch(ag, mk(p.brain,r2,s2)))
        return new
    elif algo=="elitism":
        new=[]
        for i,ag in enumerate(agents):
            if i<min(2,n): new.append(_ch(ag,sc[i].brain.clone()))
            else: new.append(_ch(ag, mk(sc[i%min(2,n)].brain,r,s)))
        return new
    elif algo=="mentor":
        mentor=sc[0].brain; new=[]
        for i,ag in enumerate(agents):
            if i==0: new.append(_ch(ag,mentor.clone()))
            else:
                nb=ag.brain.clone(); nb.blend(mentor,0.4); nb.mutate(r,max(0.07,0.35-gen*0.005))
                new.append(_ch(ag,nb))
        return new
    elif algo=="annealing":
        t=max(0.05,math.exp(-gen*0.04))
        return [_ch(ag, mk(sc[0].brain,min(0.6,t+0.15),min(0.8,t*1.5+0.1))) for ag in agents]
    elif algo=="ppo_lite":
        new=[]
        for ag in agents:
            ag.pbuf.update(ag.brain,0.004,4); ag.pbuf.clear()
            nb=ag.brain.clone(); nb.mutate(0.08,0.10); c=_ch(ag,nb); c.pbuf=PPOBuffer(); new.append(c)
        return new
    elif algo=="dqn_lite":
        return [_ch(ag, ag.brain.clone() if isinstance(ag.brain,DQNBrain) else DQNBrain(ag.brain.inp)) for ag in agents]
    elif algo=="marl":
        ts={};tc={}
        for ag in agents: ts[ag.team]=ts.get(ag.team,0)+ag.score; tc[ag.team]=tc.get(ag.team,0)+1
        bt=max({t:ts[t]/tc[t] for t in ts},key=lambda t:{t:ts[t]/tc[t] for t in ts}[t])
        best=max([a for a in agents if a.team==bt],key=lambda a:a.score)
        new=[]
        for ag in agents:
            if ag.team==bt and ag is best: new.append(_ch(ag,best.brain.clone()))
            else: new.append(_ch(ag, mk(best.brain,r,max(0.07,0.32-gen*0.005))))
        return new
    elif algo=="perfection":
        best = sc[0]
        new = []
        for i, ag in enumerate(agents):
            if i == 0:
                new.append(_ch(ag, best.brain.clone()))
            else:
                nb = best.brain.clone(); nb.mutate(0.10, 0.12)
                new.append(_ch(ag, nb))
        return new
    elif algo in ("boost", "cheat", "god"):
        # Hızlandırılmış modlar: neuroevolution gibi ama daha agresif mutasyon
        best = sc[0].brain
        new = []
        for ag in agents:
            nb = best.clone(); nb.mutate(0.30, 0.35)
            new.append(_ch(ag, nb))
        return new
    return agents

# ── Imitation Learning ────────────────────────────────────────────────────────
def imitation_learn(agents, player, epochs=5, lr=0.005):
    """Agents learn from human player's recorded actions."""
    if len(player.imitation_data) < 16: return 0
    data=player.imitation_data[-200:]
    S=np.array([d[0] for d in data],dtype=np.float32)
    A=np.array([d[1] for d in data],dtype=np.float32)
    trained=0
    for ag in agents:
        if isinstance(ag.brain,DQNBrain): continue
        inp_size=ag.brain.inp
        S_in=np.zeros((len(S),inp_size),dtype=np.float32)
        S_in[:,:min(7,inp_size)]=S[:,:min(7,inp_size)]
        for _ in range(epochs):
            idx=np.random.choice(len(S_in),min(32,len(S_in)),replace=False)
            s=S_in[idx]; a=A[idx]
            h=np.tanh(s@ag.brain.w1+ag.brain.b1)
            out=np.tanh(h@ag.brain.w2+ag.brain.b2)
            err=a-out; dout=err*(1-out**2)
            ag.brain.w2+=lr*(h.T@dout)/len(idx); ag.brain.b2+=lr*dout.mean(0)
            dh=(dout@ag.brain.w2.T)*(1-h**2)
            ag.brain.w1+=lr*(s.T@dh)/len(idx); ag.brain.b1+=lr*dh.mean(0)
        trained+=1
    return trained

def make_agents(lnum, lcfg, isz, algo, tb=None):
    tl=lcfg["touch_limit"]
    if lnum==5:
        setup=[(0,100,FH*0.35,BLUE_LT,BLUE_MD,BLUE_DK),(0,100,FH*0.65,BLUE_LT,BLUE_MD,BLUE_DK),
               (1,FW-100,FH*0.35,RED_LT,RED_MD,RED_DK),(1,FW-100,FH*0.65,RED_LT,RED_MD,RED_DK)]
    else:
        setup=[(0,90+i*20,FH*0.38 if i==0 else FH*0.62,BLUE_LT,BLUE_MD,BLUE_DK) for i in range(lcfg["n_blue"])]
    ags=[]
    for team,sx,sy,ct,cf,cs in setup:
        if algo=="dqn_lite": brain=DQNBrain(isz)
        elif tb is not None: brain=tb.clone(); brain.mutate(0.15,0.20)
        else: brain=Brain(isz)
        ag=Agent(brain,team,sx,sy,ct,cf,cs); ag.touch_limit=tl; ags.append(ag)
    return ags

INP_SIZE={1:7,2:7,3:9,4:11,5:13}

# ── Drawing helpers ───────────────────────────────────────────────────────────
def dtxt(t, x, y, c, f=FM, ctr=False):
    s=f.render(str(t),True,c)
    if ctr: x-=s.get_width()//2
    screen.blit(s,(int(x),int(y)))

def ap(color, alpha, pts):
    s=pygame.Surface((W,H),pygame.SRCALPHA); pygame.draw.polygon(s,(*color,alpha),pts); screen.blit(s,(0,0))

def aell(color, alpha, rect):
    s=pygame.Surface((W,H),pygame.SRCALPHA); pygame.draw.ellipse(s,(*color,alpha),rect); screen.blit(s,(0,0))

def arect(color, alpha, rect, br=0):
    s=pygame.Surface((W,H),pygame.SRCALPHA); pygame.draw.rect(s,(*color,alpha),rect,border_radius=br); screen.blit(s,(0,0))

def hline(y_pos, x0, x1, col=PBD):
    pygame.draw.line(screen, col, (x0, y_pos), (x1, y_pos), 1)

def section_label(text, x, y, w):
    """Draws a subtle section divider with label."""
    hline(y+7, x, x+w)
    bg=FS.render(text,True,GRAY)
    bx=x+(w-bg.get_width())//2
    pygame.draw.rect(screen,PBG,(bx-4,y,bg.get_width()+8,15))
    screen.blit(bg,(bx,y+1))

# ── Field ─────────────────────────────────────────────────────────────────────
def draw_field():
    pygame.draw.polygon(screen,FIELD_COL,[iso(0,0),iso(FW,0),iso(FW,FH),iso(0,FH)])
    for i in range(10):
        x0=FW/10*i; x1=FW/10*(i+1)
        pygame.draw.polygon(screen,FIELD_COL if i%2==0 else FIELD_COL2,
                            [iso(x0,0),iso(x1,0),iso(x1,FH),iso(x0,FH)])
    def fl(ax,ay,bx,by,a=90):
        pygame.draw.line(screen,(*WHITE,a),iso(ax,ay),iso(bx,by),2)
    fl(0,0,FW,0); fl(0,FH,FW,FH); fl(0,0,0,FH); fl(FW,0,FW,FH); fl(FW//2,0,FW//2,FH)
    cpts=[iso(FW/2+math.cos(2*math.pi*i/48)*70,FH/2+math.sin(2*math.pi*i/48)*70) for i in range(48)]
    pygame.draw.lines(screen,(*WHITE,65),True,cpts,2)

def draw_goal(gx,gy1,gy2,depth=GD,height=GH3,left=False):
    d=-depth if left else depth; NC=(175,195,235)
    ap(NC,25,[iso(gx,gy1),iso(gx,gy2),iso(gx+d,gy2),iso(gx+d,gy1)])
    for fy in [gy1,gy2]:
        for fdx in [0,d]: pygame.draw.line(screen,WHITE,iso(gx+fdx,fy,0),iso(gx+fdx,fy,height),4)
    for ax,ay,az,bx,by,bz in [
        (gx,gy1,0,gx,gy2,0),(gx,gy1,height,gx,gy2,height),
        (gx+d,gy1,0,gx+d,gy2,0),(gx+d,gy1,height,gx+d,gy2,height),
        (gx,gy1,0,gx+d,gy1,0),(gx,gy2,0,gx+d,gy2,0),
        (gx,gy1,height,gx+d,gy1,height),(gx,gy2,height,gx+d,gy2,height)]:
        pygame.draw.line(screen,WHITE,iso(ax,ay,az),iso(bx,by,bz),3)
    for i in range(8):
        t=i/7; fy=gy1+(gy2-gy1)*t
        pygame.draw.line(screen,(*NC,45),iso(gx,fy,0),iso(gx,fy,height),1)
        pygame.draw.line(screen,(*NC,25),iso(gx,fy,0),iso(gx+d,fy,0),1)
    for j in range(6):
        fz=height*j/5
        pygame.draw.line(screen,(*NC,45),iso(gx,gy1,fz),iso(gx,gy2,fz),1)

def draw_cube(x,y,sz,h3,bz,ct,cf,cs,flash,fc,sq,tn,gk=False):
    sx2=sz*sq
    offs=[(-1,-1),(1,-1),(1,1),(-1,1)]
    bot=[iso(x+o[0]*sx2,y+o[1]*sx2*0.5,bz) for o in offs]
    top=[iso(x+o[0]*sx2,y+o[1]*sx2*0.5,h3+bz) for o in offs]
    sx0,sy0=iso(x,y,0)
    aell((0,0,0),45,(sx0-int(sx2*1.4),sy0-int(sx2*0.35),int(sx2*2.8),int(sx2*0.70)))
    c0=fc if flash>0 else ct; c1=fc if flash>0 else cf; c2=fc if flash>0 else cs
    pygame.draw.polygon(screen,c2,[bot[0],bot[3],top[3],top[0]])
    pygame.draw.polygon(screen,c1,[bot[1],bot[2],top[2],top[1]])
    pygame.draw.polygon(screen,c0,[top[0],top[1],top[2],top[3]])
    ap2=bot+top
    for a2,b2 in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        pygame.draw.line(screen,(0,0,0),ap2[a2],ap2[b2],1)
    ex,ey=iso(x+sx2*0.55,y-sx2*0.35,h3*0.72+bz)
    pygame.draw.circle(screen,WHITE,(ex,ey),5); pygame.draw.circle(screen,(20,20,60),(ex+1,ey),2)
    pygame.draw.arc(screen,WHITE,pygame.Rect(ex-5,ey+4,10,6),math.pi,2*math.pi,2)
    if gk:
        kx,ky=iso(x,y,h3+bz+5); pygame.draw.ellipse(screen,GOLD,(kx-int(sx2*1.1),ky-4,int(sx2*2.2),8))
    lx,ly=iso(x,y,h3+bz+9); dtxt(str(tn+1),lx-4,ly-8,WHITE,FXS)

def draw_agent(ag):
    bz=math.sin(ag.bob)*3
    draw_cube(ag.x,ag.y,ag.SZ,ag.H3,bz,ag.ct,ag.cf,ag.cs,ag.flash,ag.fc,ag.squish,ag.team)

def draw_gk(gk):
    bz=math.sin(gk.bob)*3
    draw_cube(gk.x,gk.y,gk.SZ,gk.H3,bz,gk.ct,gk.cf,gk.cs,gk.flash,gk.fc,1.0,gk.team,True)

def draw_human(pl):
    if not pl: return
    bz=math.sin(pl.bob)*3
    c=pl.fc if pl.flash>0 else PLAYER_C
    draw_cube(pl.x,pl.y,pl.SZ,pl.H3,bz,c,(200,180,40),(160,140,20),pl.flash,pl.fc,pl.squish,2)

def draw_ball(ball):
    bx,by=iso(ball.x,ball.y,ball.R); r=11; sx,sy=iso(ball.x,ball.y,0)
    aell((0,0,0),58,(sx-r,sy-r//3,r*2,max(r//2,4)))
    pygame.draw.circle(screen,(240,240,240),(bx,by),r)
    for i in range(5):
        a2=ball.angle+i*math.pi*2/5
        px=bx+int(math.cos(a2)*r*0.5); py=by+int(math.sin(a2)*r*0.5)
        pygame.draw.circle(screen,(30,30,30),(px,py),int(r*0.28))
    pygame.draw.circle(screen,(80,80,80),(bx,by),r,2)

def draw_goal_flash(timer,team=0):
    if timer<=0: return
    col=BLUE_LT if team==0 else RED_LT
    arect(col,min(timer*5,95),pygame.Rect(0,0,FIELD_AREA_W,H))
    scale=1.0+(70-timer)*0.022; base=FB.render("GOAL!!!",True,GOLD)
    tw,th=base.get_size()
    s=pygame.transform.scale(base,(int(tw*scale),int(th*scale)))
    screen.blit(s,(FIELD_AREA_W//2-s.get_width()//2,H//2-s.get_height()//2-50))

# ── RIGHT PANEL ───────────────────────────────────────────────────────────────
def draw_panel(state):
    ags=state["ags"]; gk=state["gk"]; gen=state["gen"]; tg=state["tg"]
    bs=state["bs"];   sh=state["sh"]; lnum=state["lnum"]; lcfg=state["lcfg"]
    algo=state["algo"]; spd=state["spd"]; etick=state["etick"]; emax=state["emax"]
    sb=state["sb"];   sr2=state["sr"]; player=state["player"]
    player_on=state["player_on"]; imitation_n=state["imitation_n"]

    PX=FIELD_AREA_W; PY=0; PW=PANEL_W; PH=H

    # Background
    pygame.draw.rect(screen,PBG,(PX,PY,PW,PH))
    pygame.draw.line(screen,PBD,(PX,0),(PX,H),2)

    y=8
    # Title
    dtxt("AI FOOTBALL",PX+PW//2,y,GOLD,FB,ctr=True); y+=28

    # Level badge
    arect(lcfg.get("color",GREEN),190,(PX+8,y,PW-16,22),br=5)
    dtxt(lcfg["name"],PX+PW//2,y+3,BG,FXS,ctr=True); y+=28

    # Algorithm badge
    ac=ALGORITHMS.get(algo,{}).get("color",CYAN) if isinstance(ALGORITHMS.get(algo),dict) else CYAN
    if not isinstance(ac,tuple): ac=CYAN
    arect(ac,140,(PX+8,y,PW-16,20),br=4)
    short=ALGORITHMS.get(algo,{}).get("short","?") if isinstance(ALGORITHMS.get(algo),dict) else "?"
    dtxt(f"Algorithm: {short}",PX+PW//2,y+2,BG,FXS,ctr=True); y+=26

    hline(y,PX+8,PX+PW-8); y+=8

    # Stats grid
    col_r=PX+PW-14
    for label,val,col in [("Generation",str(gen),CYAN),("Goals",str(tg),GOLD),
                           ("Best score",f"{bs:.0f}",GREEN),("Speed",f"{spd}x",WHITE)]:
        dtxt(label,PX+12,y,GRAY,FXS)
        sv=FM.render(str(val),True,col)
        screen.blit(sv,(col_r-sv.get_width(),y-2))
        y+=20

    hline(y,PX+8,PX+PW-8); y+=6

    # Episode bar
    bw=PW-24
    dtxt("Episode",PX+12,y,GRAY,FXS)
    sec_left=(emax-etick)//60
    dtxt(f"{sec_left}s left",col_r-FM.render(f"{sec_left}s left",True,WHITE).get_width(),y-1,GRAY,FXS)
    y+=15
    ratio=min(etick/max(emax,1),1.0)
    pygame.draw.rect(screen,DGRAY,(PX+12,y,bw,10),border_radius=4)
    pygame.draw.rect(screen,CYAN,(PX+12,y,int(bw*ratio),10),border_radius=4)
    y+=16

    if lcfg["match_mode"]:
        hline(y,PX+8,PX+PW-8); y+=6
        dtxt(f"Blue  {sb}  —  {sr2}  Red",PX+PW//2,y,GOLD,FM,ctr=True); y+=22

    # Human player info
    # Goal objective tracker
    hline(y,PX+8,PX+PW-8); y+=6
    fgg=state.get("first_goal_gen",None)
    if fgg is None:
        arect(ORANGE,120,(PX+8,y,PW-16,22),br=5)
        dtxt("OBJECTIVE: Score first goal!",PX+PW//2,y+3,BG,FXS,ctr=True); y+=28
    else:
        arect(GREEN,140,(PX+8,y,PW-16,22),br=5)
        dtxt(f"GOAL! First scored gen {fgg}  Total: {state['tg']}",PX+PW//2,y+3,BG,FXS,ctr=True); y+=28

    if player_on and player:
        hline(y,PX+8,PX+PW-8); y+=6
        arect(PLAYER_C,80,(PX+8,y,PW-16,18),br=4)
        dtxt("HUMAN PLAYER  (WASD + Space)",PX+PW//2,y+2,(30,20,0),FXS,ctr=True); y+=22
        dtxt(f"Recorded moves: {len(player.imitation_data)}",PX+12,y,GRAY,FXS)
        if imitation_n>0: dtxt(f"Imitation: trained {imitation_n} agents",PX+12,y+13,GREEN,FXS)
        y+=28

    hline(y,PX+8,PX+PW-8); y+=6

    # Agents
    for ag in ags:
        col=BLUE_LT if ag.team==0 else RED_LT
        sc_col=GREEN if ag.score>=0 else RED
        dtxt(f"Agent T{ag.team+1}",PX+12,y,col,FS)
        sv=FM.render(f"{ag.score:.0f}",True,sc_col)
        screen.blit(sv,(col_r-sv.get_width(),y-1))
        y+=17

        # Touch timer bar
        tr=min(ag.touch_timer/max(ag.touch_limit*60,1),1.0)
        tc=GREEN if tr<0.5 else (ORANGE if tr<0.85 else RED)
        pygame.draw.rect(screen,DGRAY,(PX+12,y,bw,7),border_radius=3)
        if tr>0: pygame.draw.rect(screen,tc,(PX+12,y,int(bw*tr),7),border_radius=3)
        dtxt(f"No touch: {ag.touch_timer/60:.1f}s",PX+12,y+8,tc,FXS); y+=19

        # Ball-to-goal bar
        if ag.dd and ag.tracking:
            dr=1.0-min(ag.dd/math.hypot(FW,FH),1.0)
            dc=GREEN if dr>0.6 else (ORANGE if dr>0.3 else RED)
            pygame.draw.rect(screen,DGRAY,(PX+12,y,bw,7),border_radius=3)
            pygame.draw.rect(screen,dc,(PX+12,y,int(bw*dr),7),border_radius=3)
            dtxt(f"Ball-to-goal: {ag.dd:.0f}",PX+12,y+8,dc,FXS); y+=19
        else:
            dtxt("Ball-to-goal: —",PX+12,y+8,DGRAY,FXS); y+=19

        if isinstance(ag.brain,DQNBrain):
            dtxt(f"ε = {ag.brain.eps:.2f}",PX+12,y,PURPLE,FXS); y+=13

        for msg,mc,_ in ag.ev[-2:]: dtxt(msg,PX+14,y,mc,FXS); y+=12
        y+=2
        hline(y,PX+8,PX+PW-8); y+=5

    # Score history chart
    if len(sh)>=2:
        section_label("Score History",PX+8,y,PW-16); y+=16
        gh=60; gw=PW-24
        pygame.draw.rect(screen,DGRAY,(PX+12,y,gw,gh),border_radius=4)
        mn=min(sh); mx=max(sh); rng=max(mx-mn,1); hist=sh[-gw:]
        pts=[(PX+12+int(ii*gw/max(len(hist)-1,1)),y+gh-int((sc-mn)/rng*(gh-4))-2) for ii,sc in enumerate(hist)]
        if len(pts)>=2: pygame.draw.lines(screen,GREEN,False,pts,2)
        # zero line
        if mn<0<mx:
            zero_y=y+gh-int((0-mn)/rng*(gh-4))-2
            pygame.draw.line(screen,(*GRAY,80),(PX+12,zero_y),(PX+12+gw,zero_y),1)
        y+=gh+8

    # Controls
    section_label("Controls",PX+8,y,PW-16); y+=16
    ctrls=[
        ("LEFT / RIGHT","Sim speed"),
        ("UP / DOWN","Agents"),
        ("1 – 5","Level (brain kept)"),
        ("A","Algorithm menu"),
        ("S","Settings / Rewards"),
        ("P","Toggle player"),
        ("I","Imitation learn"),
        ("O","Objectives"),
        ("SPACE","Pause"),
        ("R","Reset"),
        ("M","Mutate"),
        ("ESC","Quit"),
    ]
    for k,d in ctrls:
        if y>PH-14: break
        ks=FXS.render(k,True,CYAN); ds=FXS.render(d,True,GRAY)
        screen.blit(ks,(PX+12,y))
        screen.blit(ds,(col_r-ds.get_width(),y))
        y+=13

# ── SETTINGS MENU (reward editor) ────────────────────────────────────────────
def draw_settings(lcfg, lnum, selected_row):
    MW=580; MH=580; MX=FIELD_AREA_W//2-MW//2; MY=H//2-MH//2
    arect(PBG,230,pygame.Rect(MX,MY,MW,MH),br=14)
    pygame.draw.rect(screen,PBD,pygame.Rect(MX,MY,MW,MH),2,border_radius=14)

    dtxt("SETTINGS — REWARD EDITOR",MX+MW//2,MY+12,GOLD,FB,ctr=True)
    dtxt(f"Level {lnum}: {lcfg['name']}",MX+MW//2,MY+38,GRAY,FXS,ctr=True)
    dtxt("UP/DOWN select  •  LEFT/RIGHT adjust  •  S to close",MX+MW//2,MY+52,GRAY,FXS,ctr=True)

    reward=lcfg["reward"]
    keys=list(reward.keys())
    labels={
        "touch":              "Touch reward",
        "no_touch_tick":      "No-touch penalty / tick",
        "ball_to_goal":       "Ball-to-goal multiplier",
        "goal":               "Goal reward",
        "timeout":            "Timeout penalty",
        "proximity":          "Proximity to ball bonus",
        "dribble_max":        "Max dribble bonus",
        "approach_bonus":     "Approach ball bonus",
        "goal_approach_mult": "Goal approach multiplier",
        "consecutive_touch":  "Consecutive touch bonus",
        "wall_penalty":       "Near wall penalty",
        "idle_penalty":       "Idle (no movement) penalty",
        "speed_bonus":        "Movement speed bonus",
        "center_field_bonus": "Center field bonus",
        "pass_bonus":         "Team pass bonus",
        "near_goal_bonus":    "Ball near goal bonus",
        "shot_on_target":     "Shot on target bonus",
        "touch_limit":        "Touch time limit (s)",
        "episode_s":          "Episode duration (s)",
    }
    all_keys=keys+["touch_limit","episode_s"]
    extra={"touch_limit":lcfg["touch_limit"],"episode_s":lcfg["episode_s"]}

    y=MY+72
    for i,k in enumerate(all_keys):
        val=reward.get(k,extra.get(k,0))
        is_sel=(i==selected_row)
        if is_sel: arect(CYAN,40,pygame.Rect(MX+10,y-2,MW-20,22),br=4)
        lbl=labels.get(k,k)
        col=WHITE if is_sel else GRAY
        dtxt(lbl,MX+20,y,col,FS)
        vc=GOLD if is_sel else WHITE
        dtxt(f"{val:.3f}" if isinstance(val,float) else str(val),MX+MW-60,y,vc,FM)
        if is_sel:
            dtxt("◄",MX+MW-110,y,CYAN,FM); dtxt("►",MX+MW-25,y,CYAN,FM)
        y+=26
    return all_keys

# ── ALGORITHM MENU ────────────────────────────────────────────────────────────
def draw_algo_menu(current):
    MW=600; MH=min(H-40, 56*len(ALGORITHMS)+80); MX=FIELD_AREA_W//2-MW//2; MY=max(10,H//2-MH//2)
    arect(PBG,225,pygame.Rect(MX,MY,MW,MH),br=14)
    pygame.draw.rect(screen,PBD,pygame.Rect(MX,MY,MW,MH),2,border_radius=14)
    dtxt("SELECT ALGORITHM",MX+MW//2,MY+12,GOLD,FB,ctr=True)
    dtxt("Press 1–8 to select  •  A to close",MX+MW//2,MY+34,GRAY,FXS,ctr=True)
    for i,(key,alg) in enumerate(ALGORITHMS.items()):
        row_h=min(56,(MH-70)//max(len(ALGORITHMS),1)); ry=MY+58+i*row_h; isc=(key==current)
        col=alg.get("color",CYAN) if isinstance(alg,dict) else CYAN
        if not isinstance(col,tuple): col=CYAN
        arect(col if isc else (38,48,76), 125 if isc else 50, pygame.Rect(MX+10,ry,MW-20,row_h-6),br=8)
        pygame.draw.rect(screen,col if isc else PBD,pygame.Rect(MX+10,ry,MW-20,row_h-6),2,border_radius=8)
        arect(col,195,pygame.Rect(MX+16,ry+9,44,30),br=5)
        short=alg.get("short","?") if isinstance(alg,dict) else "?"
        name=alg.get("name","?") if isinstance(alg,dict) else "?"
        desc=alg.get("desc","") if isinstance(alg,dict) else ""
        dtxt(short,MX+38,ry+15,BG,FS,ctr=True)
        dtxt(name,MX+68,ry+8,WHITE if isc else GRAY,FS)
        dtxt(desc,MX+68,ry+25,GRAY,FXS)
        key_label = str(i+1) if i<9 else "0"
        dtxt(f"[{key_label}]",MX+MW-30,ry+8,CYAN,FXS)

# ── OBJECTIVES PANEL ─────────────────────────────────────────────────────────
def draw_objectives(obj_progress):
    MW=420; MH=300; MX=FIELD_AREA_W//2-MW//2; MY=H//2-MH//2
    arect(PBG,225,pygame.Rect(MX,MY,MW,MH),br=12)
    pygame.draw.rect(screen,PBD,pygame.Rect(MX,MY,MW,MH),2,border_radius=12)
    dtxt("OBJECTIVES",MX+MW//2,MY+10,GOLD,FB,ctr=True)
    dtxt("O to close",MX+MW//2,MY+32,GRAY,FXS,ctr=True)
    y=MY+52
    for key,obj in OBJECTIVES.items():
        done=obj_progress.get(key,False)
        col=GREEN if done else GRAY
        mark="✓" if done else "○"
        dtxt(f"{mark}  {obj['label']}",MX+20,y,col,FM); y+=28
        if obj.get("bonus",0)>0:
            dtxt(f"     Bonus: +{obj['bonus']} pts",MX+20,y,DGRAY,FXS); y+=18
        else:
            y+=2
    if not OBJECTIVES:
        dtxt("No objectives defined in config.py",MX+MW//2,MY+MH//2,GRAY,FS,ctr=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    global current_algo
    current_algo="neuroevolution"

    lnum=1; lcfg=copy.deepcopy(LEVELS[1]); isz=INP_SIZE[1]; algo=current_algo
    best_brain=None

    ags=make_agents(lnum,lcfg,isz,algo,None)
    gk=Goalkeeper() if lcfg["has_goalkeeper"] else None
    ball=Ball()
    player=HumanPlayer()

    gen=1; tg=0; bs=0.0; sh=[]; emax=lcfg["episode_s"]*60; etick=0
    gf=0; gt=0; sb=0; sr_score=0
    first_goal_gen=None   # which generation scored first goal
    sopts=[1,2,3,5,8,10,15,30,60,100,200,500]; si=2; spd=3
    paused=False; smenu=False; show_algo=False; show_obj=False
    player_on=False; imitation_n=0
    settings_row=0; settings_keys=[]
    obj_progress={k:False for k in OBJECTIVES}

    def rep():
        nonlocal etick
        etick=0; ball.reset()
        for ag in ags: ag.reset(); ag.touch_limit=lcfg["touch_limit"]
        if gk: gk.reset()

    def svb():
        nonlocal best_brain
        if ags:
            b=max(ags,key=lambda a:a.score)
            if not isinstance(b.brain,DQNBrain): best_brain=b.brain.clone()

    def ll(n,kb=True):
        nonlocal lnum,lcfg,isz,ags,gk,ball,algo,gen,tg,bs,sh,emax,etick,gf,sb,sr_score,obj_progress,first_goal_gen
        svb(); lnum=n; lcfg=copy.deepcopy(LEVELS[n]); isz=INP_SIZE[n]; algo=current_algo
        tb=best_brain if kb else None
        ags=make_agents(n,lcfg,isz,algo,tb)
        gk=Goalkeeper() if lcfg["has_goalkeeper"] else None
        ball=Ball(); gen=1; tg=0; bs=0.0; sh.clear()
        emax=lcfg["episode_s"]*60; etick=0; gf=0; sb=0; sr_score=0
        obj_progress={k:False for k in OBJECTIVES}; first_goal_gen=None

    running=True
    while running:
        keys=pygame.key.get_pressed()

        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: running=False
            if ev.type==pygame.KEYDOWN:
                if ev.key==pygame.K_ESCAPE:
                    if smenu or show_algo or show_obj: smenu=show_algo=show_obj=False
                    else: running=False
                # P always works regardless of player mode
                if ev.key==pygame.K_p:
                    player_on=not player_on
                    if not player_on: player.reset()
                # When player is ON, WASD/Space are for movement — block other hotkeys
                if not player_on:
                    if ev.key==pygame.K_SPACE and not smenu and not show_algo:
                        paused=not paused
                    if ev.key==pygame.K_a and not smenu: show_algo=not show_algo; show_obj=False
                    if ev.key==pygame.K_s: smenu=not smenu; show_algo=False; show_obj=False
                    if ev.key==pygame.K_o: show_obj=not show_obj; show_algo=False; smenu=False
                    if ev.key==pygame.K_i:
                        imitation_n=imitation_learn(ags,player)
                if not player_on and not smenu and not show_algo:
                    if ev.key==pygame.K_RIGHT: si=min(si+1,len(sopts)-1); spd=sopts[si]
                    if ev.key==pygame.K_LEFT:  si=max(si-1,0); spd=sopts[si]
                    if ev.key==pygame.K_UP and lnum<5:
                        if len([a for a in ags if a.team==0])<6:
                            sy2=FH*0.4+len(ags)*40
                            nb=DQNBrain(isz) if algo=="dqn_lite" else Brain(isz)
                            ag2=Agent(nb,0,90,sy2,BLUE_LT,BLUE_MD,BLUE_DK)
                            ag2.touch_limit=lcfg["touch_limit"]; ags.append(ag2)
                    if ev.key==pygame.K_DOWN and len(ags)>1:
                        bl=[a for a in ags if a.team==0]
                        if len(bl)>1: ags.remove(bl[-1])
                    if ev.key==pygame.K_r: ll(lnum,kb=False)
                    if ev.key==pygame.K_m:
                        for ag in ags:
                            if not isinstance(ag.brain,DQNBrain): ag.brain.mutate(0.65,0.55)
                if smenu:
                    n_rows=len(lcfg["reward"])+2
                    if ev.key==pygame.K_DOWN: settings_row=(settings_row+1)%n_rows
                    if ev.key==pygame.K_UP:   settings_row=(settings_row-1)%n_rows
                    if ev.key==pygame.K_RIGHT or ev.key==pygame.K_LEFT:
                        direction=1 if ev.key==pygame.K_RIGHT else -1
                        if settings_keys and settings_row<len(settings_keys):
                            k=settings_keys[settings_row]
                            if k in lcfg["reward"]:
                                v=lcfg["reward"][k]
                                step=1.0 if abs(v)>5 else (0.01 if abs(v)<0.5 else 0.1)
                                lcfg["reward"][k]=round(v+direction*step,3)
                            elif k=="touch_limit":
                                lcfg["touch_limit"]=max(1,lcfg["touch_limit"]+direction)
                            elif k=="episode_s":
                                lcfg["episode_s"]=max(5,lcfg["episode_s"]+direction*5)
                                emax=lcfg["episode_s"]*60
                # 1-9 ve 0 tuşları (0 = 10. algoritma)
                algo_nums = list(range(1,10)) + [0]
                for num_idx, num in enumerate(algo_nums):
                    k2=getattr(pygame,f"K_{num}",None)
                    algo_index = num_idx  # 0-9 arası
                    if k2 and ev.type==pygame.KEYDOWN and ev.key==k2:
                        if show_algo:
                            if algo_index<len(ALGO_KEYS):
                                current_algo=ALGO_KEYS[algo_index]; algo=current_algo; show_algo=False
                                for ag in ags:
                                    if algo=="dqn_lite" and not isinstance(ag.brain,DQNBrain): ag.brain=DQNBrain(isz)
                                    elif algo!="dqn_lite" and isinstance(ag.brain,DQNBrain): ag.brain=Brain(isz)
                        elif not smenu:
                            if 1<=num<=5 and num!=lnum: ll(num,kb=True)
                            # 0 tuşu level seçimi için değil

        if paused and not smenu and not show_algo and not show_obj:
            screen.fill(BG)
            pygame.draw.rect(screen,PBG,(FIELD_AREA_W,0,PANEL_W,H))
            pygame.draw.line(screen,PBD,(FIELD_AREA_W,0),(FIELD_AREA_W,H),2)
            dtxt("PAUSED  —  SPACE to resume",FIELD_AREA_W//2,H//2-12,GOLD,FB,ctr=True)
            state={"ags":ags,"gk":gk,"gen":gen,"tg":tg,"bs":bs,"sh":sh,"lnum":lnum,
                   "lcfg":lcfg,"algo":algo,"spd":spd,"etick":etick,"emax":emax,
                   "sb":sb,"sr":sr_score,"player":player,"player_on":player_on,"imitation_n":imitation_n,
                   "first_goal_gen":first_goal_gen}
            draw_panel(state)
            pygame.display.flip(); clock.tick(30); continue

        # ── Simulation ────────────────────────────────────────────────────────
        for _ in range(spd):
            ball.step()
            if gk: gk.step(ball); gk.block(ball)
            if player_on: player.update(ball,keys)
            for ag in ags:
                mts=[a for a in ags if a is not ag and a.team==ag.team]
                ops=[a for a in ags if a.team!=ag.team]
                ag.step(ball,lcfg["reward"],algo,
                        gk=gk if ag.team==0 else None,mates=mts or None,opps=ops or None)
                ag.te()
            etick+=1
            if algo=="marl":
                for tid in [0,1]:
                    tm=[a for a in ags if a.team==tid]
                    if len(tm)>1:
                        avg=sum(a.score for a in tm)/len(tm)
                        for a in tm: a.score=avg

            def hg(st):
                nonlocal tg,gf,gt,sb,sr_score
                tg+=1; gf=70; gt=st; bonus=lcfg["reward"]["goal"]
                for ag in ags:
                    if ag.team==st: ag.score+=bonus; ag.goals+=1
                if st==0: sb+=1; obj_progress["score_goal"]=True
                if first_goal_gen is None: first_goal_gen=gen
                else: sr_score+=1
                ball.reset()
                for ag in ags: ag.touch_timer=0; ag.tracking=False

            if ball.in_right():
                st=ball.last_toucher if ball.last_toucher is not None else 0
                if st==0 or not lcfg["match_mode"]: hg(0)
                else: ball.reset()
            if lcfg["match_mode"] and ball.in_left():
                st=ball.last_toucher if ball.last_toucher is not None else 1
                hg(1 if st==1 else 0)
            if gf>0: gf-=1

            # BOOST: episode 3x uzun
            effective_emax = emax * 3 if algo=="boost" else emax
            if etick>=effective_emax:
                svb()
                for ag in ags:
                    if ag.score>bs: bs=ag.score
                    sh.append(ag.score)
                ags=evolve_agents(ags,gen,algo)
                for ag in ags: ag.touch_limit=lcfg["touch_limit"]
                gen+=1; rep()

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG)

        # Field area clip
        pygame.draw.rect(screen,BG,(0,0,FIELD_AREA_W,H))
        draw_field()
        draw_goal(GX,GY1,GY2)
        if lcfg["match_mode"]: draw_goal(LGX,LGY1,LGY2,left=True)
        draw_ball(ball)
        if gk: draw_gk(gk)
        for ag in ags: draw_agent(ag)
        if player_on: draw_human(player)
        draw_goal_flash(gf,gt)

        # Player mode banner
        if player_on:
            arect(PLAYER_C,200,pygame.Rect(6,H-68,FIELD_AREA_W-12,28),br=6)
            dtxt("  PLAYER MODE ON  —  WASD: move  |  Space: kick  |  P: exit player mode  |  I: teach AI",10,H-62,(30,20,0),FXS)
        # Bottom bar inside field area
        arect(lcfg.get("color",GREEN),170,pygame.Rect(6,H-34,FIELD_AREA_W-12,28),br=6)
        dtxt(f"  {lcfg['name']}  |  {lcfg['desc']}  |  Algo: {ALGORITHMS.get(algo,{}).get('short',algo) if isinstance(ALGORITHMS.get(algo),dict) else algo}",
             10,H-28,(8,18,8),FXS)

        # Right panel
        state={"ags":ags,"gk":gk,"gen":gen,"tg":tg,"bs":bs,"sh":sh,"lnum":lnum,
               "lcfg":lcfg,"algo":algo,"spd":spd,"etick":etick,"emax":emax,
               "sb":sb,"sr":sr_score,"player":player,"player_on":player_on,"imitation_n":imitation_n,
               "first_goal_gen":first_goal_gen}
        draw_panel(state)

        # Overlays
        if show_algo: draw_algo_menu(algo)
        if show_obj:  draw_objectives(obj_progress)
        if smenu:     settings_keys=draw_settings(lcfg,lnum,settings_row)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit(); sys.exit()

if __name__=="__main__":
    main()
