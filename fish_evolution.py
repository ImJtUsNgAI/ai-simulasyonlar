"""
Balik Surüsü + Köpekbaligi - Evrim Simülasyonu
===============================================
pip install pygame numpy
python fish_evolution.py

KONTROLLER
  SPACE     : Duraklat
  LEFT/RIGHT: Hız
  R         : Sifirla
  S         : İstatistik paneli aç/kapat
  ESC       : Çıkış

NASIL ÇALIŞIR
  - Her baligin 3 geni var: HIZ, GÖRÜŞ AÇISI, SÜRÜ CAZIBESI
  - Köpekbaligi yakalanabilen baligi yer → o gen gider
  - Hayatta kalan balıklar ürer → genler aktarılır + küçük mutasyon
  - Nesiller geçtikçe balıklar GERÇEKTEN evrimleşir
  - Sürü davranışı öğrenilir çünkü sürüdeki balık daha az yakalanır
"""

import pygame, sys, math, random
import numpy as np

pygame.init()
W, H = 1300, 780
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Balik Sürüsü Evrimi")
clock = pygame.time.Clock()

FB  = pygame.font.SysFont("segoeui", 22, bold=True)
FM  = pygame.font.SysFont("segoeui", 15)
FS  = pygame.font.SysFont("segoeui", 13)
FXS = pygame.font.SysFont("segoeui", 11)

# ── Renkler ────────────────────────────────────────────────────────────────────
WATER_DEEP   = (8,  32,  68)
WATER_MID    = (12, 48,  95)
WATER_LIGHT  = (18, 65, 120)
WHITE        = (255,255,255)
GOLD         = (255,210, 50)
RED          = (220, 55, 55)
GREEN        = ( 50,215,110)
CYAN         = ( 60,210,210)
ORANGE       = (255,150, 30)
GRAY         = (130,140,160)
DGRAY        = ( 35, 45, 70)
PANEL_BG     = ( 12, 20, 42)
PANEL_BD     = ( 45, 65,120)

# Alan boyutları
SIM_W = 950   # simülasyon alanı
PANEL_W = W - SIM_W

# ── Gen sabitleri ──────────────────────────────────────────────────────────────
# Her baligin genleri: [hiz, görüş_açısı, sürü_cazibesi]
# Hız: 1.0-4.0 (yüksek = hızlı ama göze çarpar)
# Görüş açısı: 0.3-2.5 rad (geniş = tehlikeyi erken görür)
# Sürü cazibi: 0.0-1.0 (yüksek = sürüye yakın durur)

MIN_SPEED   = 0.8
MAX_SPEED   = 4.2
MIN_FOV     = 0.4   # radyan
MAX_FOV     = 2.8
MIN_FLOCK   = 0.0
MAX_FLOCK   = 1.0

FISH_COUNT  = 120
SHARK_COUNT = 3
REPRO_RATE  = 0.004   # her tick üreme şansı (enerji yeterliyse)
MUTATION    = 0.12    # gen mutasyon gücü

# ── Balık sınıfı ──────────────────────────────────────────────────────────────
class Fish:
    R = 5   # görsel yarıçap

    def __init__(self, genes=None):
        self.x = random.uniform(30, SIM_W-30)
        self.y = random.uniform(30, H-30)
        angle  = random.uniform(0, 2*math.pi)

        if genes is None:
            self.speed     = random.uniform(MIN_SPEED, MAX_SPEED)
            self.fov       = random.uniform(MIN_FOV, MAX_FOV)
            self.flock_str = random.uniform(MIN_FLOCK, MAX_FLOCK)
        else:
            sp, fo, fl = genes
            mut = MUTATION
            self.speed     = float(np.clip(sp + random.gauss(0,mut)*1.0, MIN_SPEED, MAX_SPEED))
            self.fov       = float(np.clip(fo + random.gauss(0,mut)*0.3, MIN_FOV,   MAX_FOV))
            self.flock_str = float(np.clip(fl + random.gauss(0,mut)*0.15, MIN_FLOCK, MAX_FLOCK))

        self.vx = math.cos(angle) * self.speed
        self.vy = math.sin(angle) * self.speed
        self.energy   = 60.0
        self.age      = 0
        self.alive    = True
        self.flash    = 0   # kırmızı flash (tehlike gördü)

        # Renk: sürü cazibesine göre mavi→yeşil
        # Yüksek sürü cazibesi = daha koyu mavi (sürü rengi)
        t = self.flock_str
        self.color = (
            int(40  + (1-t)*160),
            int(140 + t*80),
            int(200 + t*55),
        )

    def get_genes(self):
        return (self.speed, self.fov, self.flock_str)

    def step(self, fish_list, sharks):
        if not self.alive: return
        self.age += 1
        self.energy -= 0.05
        if self.flash > 0: self.flash -= 1

        # ── Görüş alanı: tehlike var mı? ───────────────────────────────────
        danger_vx, danger_vy = 0.0, 0.0
        for sh in sharks:
            dx = self.x - sh.x
            dy = self.y - sh.y
            dist = math.hypot(dx, dy)
            if dist < 1: continue
            # Görüş açısı içinde mi?
            angle_to_shark = math.atan2(sh.y-self.y, sh.x-self.x)
            my_angle = math.atan2(self.vy, self.vx)
            diff = abs((angle_to_shark - my_angle + math.pi) % (2*math.pi) - math.pi)
            view_range = 200 + self.fov * 80
            if dist < view_range and diff < self.fov:
                # Kaç!
                flee_str = max(0, (view_range - dist) / view_range) * 3.0
                danger_vx += (dx/dist) * flee_str
                danger_vy += (dy/dist) * flee_str
                self.flash = 8

        # ── Sürü davranışı ─────────────────────────────────────────────────
        # Flocking: ayrılma, hizalanma, yakınlaşma
        sep_x=sep_y=0.0
        ali_x=ali_y=0.0
        coh_x=coh_y=0.0
        n_near = 0

        for other in fish_list:
            if other is self or not other.alive: continue
            dx = other.x - self.x
            dy = other.y - self.y
            dist = math.hypot(dx, dy)
            if dist < 1 or dist > 120: continue
            n_near += 1
            if dist < 25:
                sep_x -= dx/dist; sep_y -= dy/dist
            ali_x += other.vx; ali_y += other.vy
            coh_x += dx; coh_y += dy

        flock_force_x = flock_force_y = 0.0
        if n_near > 0:
            flock_force_x = sep_x*1.5 + ali_x/n_near*0.3 + coh_x/n_near*0.008
            flock_force_y = sep_y*1.5 + ali_y/n_near*0.3 + coh_y/n_near*0.008

        # ── Hız güncelleme ─────────────────────────────────────────────────
        self.vx += danger_vx + flock_force_x * self.flock_str
        self.vy += danger_vy + flock_force_y * self.flock_str

        # Hızı gene göre normalize et
        spd = math.hypot(self.vx, self.vy)
        if spd > 0.01:
            self.vx = self.vx/spd * self.speed
            self.vy = self.vy/spd * self.speed

        # ── Konum güncelle ─────────────────────────────────────────────────
        self.x += self.vx
        self.y += self.vy

        # Duvar sektir
        margin = 40
        if self.x < margin:    self.vx += 0.5
        if self.x > SIM_W-margin: self.vx -= 0.5
        if self.y < margin:    self.vy += 0.5
        if self.y > H-margin:  self.vy -= 0.5
        self.x = float(np.clip(self.x, 5, SIM_W-5))
        self.y = float(np.clip(self.y, 5, H-5))

        # Enerji tükendi → öl
        if self.energy <= 0:
            self.alive = False

    def draw(self, surf):
        if not self.alive: return
        col = (255,80,80) if self.flash>0 else self.color
        angle = math.atan2(self.vy, self.vx)
        r = self.R

        # Balık şekli: üçgen
        pts = [
            (self.x + math.cos(angle)*r*2,     self.y + math.sin(angle)*r*2),
            (self.x + math.cos(angle+2.4)*r,   self.y + math.sin(angle+2.4)*r),
            (self.x + math.cos(angle-2.4)*r,   self.y + math.sin(angle-2.4)*r),
        ]
        pygame.draw.polygon(surf, col, pts)

        # Görüş konisi (ince çizgi)
        if self.fov > 1.5:
            cone_len = 30 + self.fov*20
            cx1 = self.x + math.cos(angle+self.fov*0.5)*cone_len
            cy1 = self.y + math.sin(angle+self.fov*0.5)*cone_len
            cx2 = self.x + math.cos(angle-self.fov*0.5)*cone_len
            cy2 = self.y + math.sin(angle-self.fov*0.5)*cone_len
            pygame.draw.line(surf,(*col,60),(int(self.x),int(self.y)),(int(cx1),int(cy1)),1)
            pygame.draw.line(surf,(*col,60),(int(self.x),int(self.y)),(int(cx2),int(cy2)),1)


# ── Köpekbalığı ───────────────────────────────────────────────────────────────
class Shark:
    R = 14

    def __init__(self):
        edge = random.choice(['L','R','T','B'])
        if edge=='L': self.x,self.y = 0, random.uniform(0,H)
        elif edge=='R': self.x,self.y = SIM_W, random.uniform(0,H)
        elif edge=='T': self.x,self.y = random.uniform(0,SIM_W), 0
        else: self.x,self.y = random.uniform(0,SIM_W), H
        self.vx = random.uniform(-1,1)
        self.vy = random.uniform(-1,1)
        self.speed = 2.8
        self.energy = 100.0
        self.kills = 0
        self.target = None
        self.bob = random.uniform(0, math.pi*2)

    def step(self, fish_list):
        self.bob += 0.05
        self.energy -= 0.08

        # Hedef seç: en yakın balık
        best_dist = 999999
        self.target = None
        for f in fish_list:
            if not f.alive: continue
            d = math.hypot(f.x-self.x, f.y-self.y)
            if d < best_dist:
                best_dist = d
                self.target = f

        if self.target:
            dx = self.target.x - self.x
            dy = self.target.y - self.y
            dist = math.hypot(dx,dy)
            if dist > 0:
                self.vx += dx/dist * 0.4
                self.vy += dy/dist * 0.4

            # Yakalama
            if dist < self.R + self.target.R + 4:
                self.target.alive = False
                self.energy = min(200, self.energy + 40)
                self.kills += 1
                self.target = None

        # Hız normalize
        spd = math.hypot(self.vx,self.vy)
        if spd > 0.01:
            self.vx = self.vx/spd*self.speed
            self.vy = self.vy/spd*self.speed

        # Sınırdan dön
        if self.x < 30:   self.vx += 0.8
        if self.x > SIM_W-30: self.vx -= 0.8
        if self.y < 30:   self.vy += 0.8
        if self.y > H-30: self.vy -= 0.8
        self.x = float(np.clip(self.x+self.vx, 5, SIM_W-5))
        self.y = float(np.clip(self.y+self.vy, 5, H-5))

        if self.energy <= 0:
            self.__init__()   # köpekbalığı açlıktan ölünce yenisi gelir

    def draw(self, surf):
        angle = math.atan2(self.vy, self.vx)
        bob = math.sin(self.bob)*2
        r = self.R

        # Köpekbalığı şekli
        pts = [
            (self.x + math.cos(angle)*r*2.5,        self.y + math.sin(angle)*r*2.5 + bob),
            (self.x + math.cos(angle+2.0)*r*1.1,    self.y + math.sin(angle+2.0)*r*1.1 + bob),
            (self.x + math.cos(angle+math.pi)*r*0.8,self.y + math.sin(angle+math.pi)*r*0.8 + bob),
            (self.x + math.cos(angle-2.0)*r*1.1,    self.y + math.sin(angle-2.0)*r*1.1 + bob),
        ]
        pygame.draw.polygon(surf, (160,160,180), pts)
        pygame.draw.polygon(surf, (120,120,140), pts, 1)

        # Göz
        ex = self.x + math.cos(angle+0.4)*r*1.2
        ey = self.y + math.sin(angle+0.4)*r*1.2 + bob
        pygame.draw.circle(surf, (20,20,20), (int(ex),int(ey)), 3)
        pygame.draw.circle(surf, WHITE, (int(ex),int(ey)), 1)

        # Enerji barı
        bar_w = 30
        ex2 = int(self.x-bar_w//2)
        ey2 = int(self.y-r*2.5-8)
        ratio = min(self.energy/200,1.0)
        pygame.draw.rect(surf,DGRAY,(ex2,ey2,bar_w,5),border_radius=2)
        pygame.draw.rect(surf,RED,(ex2,ey2,int(bar_w*ratio),5),border_radius=2)


# ── Su arkaplanı ──────────────────────────────────────────────────────────────
def draw_water(surf):
    surf.fill(WATER_DEEP, (0,0,SIM_W,H))
    # Işık huzmesi efekti (statik çizgiler)
    for i in range(12):
        x = int(SIM_W/12*i + 20)
        alpha_surf = pygame.Surface((40,H), pygame.SRCALPHA)
        alpha_surf.fill((255,255,255,6))
        for j in range(0,H,8):
            offset = int(math.sin(i*1.3+j*0.02)*8)
            pygame.draw.line(alpha_surf, (200,220,255,4), (20+offset,j),(20+offset,j+6),2)
        surf.blit(alpha_surf,(x-20,0))

    # Kabarcıklar (dekoratif)
    for i in range(8):
        bx = (i*137+50) % SIM_W
        by = (i*97+30)  % H
        pygame.draw.circle(surf,(30,80,140),(bx,by),3,1)


# ── Panel ─────────────────────────────────────────────────────────────────────
def draw_panel(surf, fish_list, sharks, gen, tick, history_data, show_stats):
    PX = SIM_W; PW = PANEL_W
    pygame.draw.rect(surf, PANEL_BG, (PX,0,PW,H))
    pygame.draw.line(surf, PANEL_BD, (PX,0),(PX,H), 2)

    alive = [f for f in fish_list if f.alive]
    y = 10

    def dt(text,x,yy,col,font=FM,cx=False):
        s=font.render(str(text),True,col)
        if cx: x-=s.get_width()//2
        surf.blit(s,(int(x),int(yy)))

    def hline(yy): pygame.draw.line(surf,PANEL_BD,(PX+8,yy),(PX+PW-8,yy),1)

    dt("BALIK SÜRÜSÜ EVRİMİ",PX+PW//2,y,GOLD,FB,cx=True); y+=28

    # İstatistik kartları
    stats = [
        ("Nesil",       str(gen),          CYAN),
        ("Canlı balık", str(len(alive)),    GREEN),
        ("Köpekbalığı", str(len(sharks)),   RED),
        ("Gün",         str(tick//60),      GRAY),
    ]
    for lbl,val,col in stats:
        dt(lbl+":",  PX+12, y, GRAY, FS)
        s=FM.render(val,True,col); surf.blit(s,(PX+PW-12-s.get_width(),y-1))
        y+=20
    hline(y); y+=8

    # Ortalama gen değerleri
    if alive:
        avg_spd  = sum(f.speed     for f in alive)/len(alive)
        avg_fov  = sum(f.fov       for f in alive)/len(alive)
        avg_flk  = sum(f.flock_str for f in alive)/len(alive)

        dt("Ortalama genler:",PX+12,y,GRAY,FS); y+=16
        bw = PW-24
        for label,val,maxv,col in [
            ("Hız",        avg_spd, MAX_SPEED, (255,180,50)),
            ("Görüş açısı",avg_fov, MAX_FOV,   (80,200,200)),
            ("Sürü cazibesi",avg_flk,MAX_FLOCK,(100,200,120)),
        ]:
            ratio = val/maxv
            pygame.draw.rect(surf,DGRAY,(PX+12,y,bw,10),border_radius=3)
            pygame.draw.rect(surf,col,  (PX+12,y,int(bw*ratio),10),border_radius=3)
            dt(f"{label}: {val:.2f}",PX+12,y+12,col,FXS)
            y+=26

    hline(y); y+=8

    # Sürü boyutu dağılımı
    dt("Sürü analizi:",PX+12,y,GRAY,FS); y+=15
    # Hangi balığın kaç komşusu var
    if alive:
        flock_sizes = []
        for f in alive:
            n=sum(1 for o in alive if o is not f and math.hypot(o.x-f.x,o.y-f.y)<80)
            flock_sizes.append(n)
        avg_flock_size = sum(flock_sizes)/len(flock_sizes) if flock_sizes else 0
        max_group = max(flock_sizes) if flock_sizes else 0
        dt(f"Ort. komşu: {avg_flock_size:.1f}",PX+12,y,WHITE,FS); y+=16
        dt(f"Max grup: {max_group}",PX+12,y,CYAN,FS); y+=16

    hline(y); y+=8

    # Grafik — nesil başına ortalama hız geni
    dt("Hız geni evrimi:",PX+12,y,GRAY,FXS); y+=14
    gh=60; gw=PW-24
    pygame.draw.rect(surf,(25,35,60),(PX+12,y,gw,gh),border_radius=4)
    if len(history_data["speed"])>=2:
        hist=history_data["speed"][-gw:]
        mn=min(hist)-0.1; mx=max(hist)+0.1; rng=max(mx-mn,0.1)
        pts=[(PX+12+int(ii*gw/max(len(hist)-1,1)),
              y+gh-int((v-mn)/rng*(gh-4))-2) for ii,v in enumerate(hist)]
        if len(pts)>=2: pygame.draw.lines(surf,(255,180,50),False,pts,2)
    y+=gh+6

    dt("Sürü cazibesi evrimi:",PX+12,y,GRAY,FXS); y+=14
    pygame.draw.rect(surf,(25,35,60),(PX+12,y,gw,gh),border_radius=4)
    if len(history_data["flock"])>=2:
        hist=history_data["flock"][-gw:]
        mn=min(hist)-0.05; mx=max(hist)+0.05; rng=max(mx-mn,0.05)
        pts=[(PX+12+int(ii*gw/max(len(hist)-1,1)),
              y+gh-int((v-mn)/rng*(gh-4))-2) for ii,v in enumerate(hist)]
        if len(pts)>=2: pygame.draw.lines(surf,(100,200,120),False,pts,2)
    y+=gh+6

    hline(y); y+=6

    # Kontroller
    for k,d in [("SPACE","Duraklat"),("LEFT/RIGHT","Hız"),("R","Sıfırla"),("ESC","Çıkış")]:
        dt(k,PX+14,y,CYAN,FXS)
        s=FXS.render(d,True,GRAY); surf.blit(s,(PX+PW-12-s.get_width(),y))
        y+=13


# ── Ana döngü ─────────────────────────────────────────────────────────────────
def main():
    fish_list = [Fish() for _ in range(FISH_COUNT)]
    sharks    = [Shark() for _ in range(SHARK_COUNT)]

    generation   = 1
    tick         = 0
    sim_speed    = 3
    speed_opts   = [1,2,3,5,8,12,20]
    speed_idx    = 2
    paused       = False
    show_stats   = True

    history_data = {"speed":[], "flock":[], "fov":[]}

    # Arkaplan yüzeyi (bir kere çiz, performans için)
    bg_surf = pygame.Surface((SIM_W, H))
    draw_water(bg_surf)

    def maybe_reproduce():
        alive = [f for f in fish_list if f.alive]
        if len(alive) >= FISH_COUNT*1.5: return
        if not alive: return
        for f in alive:
            if f.energy > 70 and random.random() < REPRO_RATE:
                parent = random.choice(alive)
                child = Fish(genes=parent.get_genes())
                fish_list.append(child)

    def cleanup_and_log():
        nonlocal generation
        alive=[f for f in fish_list if f.alive]
        # Ölüleri temizle (100dan fazla birikmişse)
        dead_count=len([f for f in fish_list if not f.alive])
        if dead_count>80:
            fish_list[:]=alive

        # Log
        if alive:
            history_data["speed"].append(sum(f.speed for f in alive)/len(alive))
            history_data["flock"].append(sum(f.flock_str for f in alive)/len(alive))
            history_data["fov"].append(sum(f.fov for f in alive)/len(alive))
            if len(history_data["speed"])>300:
                for k in history_data: history_data[k].pop(0)

        # Eğer çok az balık kaldıysa yeni nesil başlat
        if len(alive)<15:
            generation+=1
            best=sorted(alive,key=lambda f:f.age,reverse=True)[:max(1,len(alive))]
            while len(alive)<FISH_COUNT:
                parent=random.choice(best)
                fish_list.append(Fish(genes=parent.get_genes()))
            alive=[f for f in fish_list if f.alive]

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
                    fish_list.clear(); fish_list.extend([Fish() for _ in range(FISH_COUNT)])
                    sharks.clear(); sharks.extend([Shark() for _ in range(SHARK_COUNT)])
                    generation=1; tick=0; history_data={k:[] for k in history_data}
                if ev.key==pygame.K_s: show_stats=not show_stats

        if paused:
            screen.blit(bg_surf,(0,0))
            for f in fish_list: f.draw(screen)
            for sh in sharks: sh.draw(screen)
            draw_panel(screen,fish_list,sharks,generation,tick,history_data,show_stats)
            s=FB.render("DURAKLADI — SPACE devam",True,GOLD)
            screen.blit(s,(SIM_W//2-s.get_width()//2,H//2-20))
            pygame.display.flip(); clock.tick(30); continue

        for _ in range(sim_speed):
            alive=[f for f in fish_list if f.alive]
            for f in alive: f.step(alive, sharks)
            for sh in sharks: sh.step(alive)
            maybe_reproduce()
            tick+=1
            if tick%120==0: cleanup_and_log()

        # ── Çizim ──────────────────────────────────────────────────────────
        screen.blit(bg_surf,(0,0))

        # Canlı balıklar (arka planda sürü bağlantıları)
        alive=[f for f in fish_list if f.alive]

        # Sürü bağlantı çizgileri
        for f in alive:
            for o in alive:
                if o is f: continue
                d=math.hypot(o.x-f.x,o.y-f.y)
                if d<45:
                    alpha=int((1-d/45)*35)
                    s2=pygame.Surface((SIM_W,H),pygame.SRCALPHA)
                    pygame.draw.line(s2,(*f.color,alpha),(int(f.x),int(f.y)),(int(o.x),int(o.y)),1)
                    screen.blit(s2,(0,0))
                    break  # performans için sadece bir bağlantı çiz

        for f in alive: f.draw(screen)
        for sh in sharks: sh.draw(screen)

        # Sürü alanı göstergesi (en büyük grup etrafında daire)
        if alive:
            cx=sum(f.x for f in alive)/len(alive)
            cy=sum(f.y for f in alive)/len(alive)
            s3=pygame.Surface((SIM_W,H),pygame.SRCALPHA)
            pygame.draw.circle(s3,(80,160,255,18),(int(cx),int(cy)),80)
            pygame.draw.circle(s3,(80,160,255,35),(int(cx),int(cy)),80,2)
            screen.blit(s3,(0,0))

        draw_panel(screen,fish_list,sharks,generation,tick,history_data,show_stats)

        # Alt bilgi
        spd_label=f"Hız: {sim_speed}x"
        s=FS.render(spd_label,True,GRAY)
        screen.blit(s,(8,H-20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__=="__main__":
    main()
