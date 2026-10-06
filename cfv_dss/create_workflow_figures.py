"""Render Indonesian research workflow diagrams as PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / 'artifacts' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans'})
COLORS = {
    'data': ('#e0f2fe', '#0284c7'),
    'process': ('#eef2ff', '#6366f1'),
    'causal': ('#dcfce7', '#16a34a'),
    'output': ('#ccfbf1', '#0d9488'),
    'note': ('#fff7ed', '#ea580c'),
}

def canvas(title, subtitle, height):
    fig, ax = plt.subplots(figsize=(14, height))
    fig.patch.set_facecolor('#ffffff')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, height)
    ax.axis('off')
    fig.subplots_adjust(left=.025, right=.975, bottom=.025, top=.975)
    ax.text(.4, height-.55, title, fontsize=21, weight='bold', color='#0f172a', va='top')
    ax.text(.4, height-1.12, subtitle, fontsize=11.5, color='#475569', va='top')
    return fig, ax

def box(ax, x, y, width, height, text, kind='process', size=11.5):
    fill, edge = COLORS[kind]
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle='round,pad=0.015,rounding_size=0.12',
                          facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=3)
    ax.add_patch(patch)
    ax.text(x+width/2, y+height/2, text, ha='center', va='center',
            fontsize=size, color='#0f172a', linespacing=1.45, zorder=4)
    return {'top': (x+width/2, y+height), 'bottom': (x+width/2, y),
            'left': (x, y+height/2), 'right': (x+width, y+height/2)}

def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=16,
                                linewidth=1.5, color='#64748b', shrinkA=5,
                                shrinkB=5, zorder=2))

def down(ax, a, b):
    arrow(ax, a['bottom'], b['top'])

def save(fig, name):
    fig.savefig(OUT / (name+'.png'), dpi=180, facecolor='white')
    plt.close(fig)
    print(name+'.png')

fig, ax = canvas('Alur riset CFV-DSS', 'Dari catatan kredit menjadi prediksi, kandidat intervensi, dan dashboard.', 14)
main = box(ax, .5, 10.8, 8.7, 1.15,
           '1. Home Credit: aplikasi + empat tabel riwayat\n307.511 pemohon | outcome: kesulitan pembayaran', 'data')
external = box(ax, 9.65, 10.8, 3.85, 1.15,
               'Give Me Some Credit\n150.000 observasi', 'data')
features = box(ax, .5, 9.05, 8.7, 1.2,
               '2. Agregasi per ID dan pembentukan 13 konstruk\nSimpan cache Parquet, profil missing, dan checksum', 'process')
down(ax, main, features)
prep = box(ax, .5, 7.25, 8.7, 1.2,
           '3. Split 64% train / 16% validation / 20% test\nFit median imputation + RobustScaler pada train', 'process')
down(ax, features, prep)
pred = box(ax, .5, 5.35, 4.1, 1.3,
           '4A. Jalur prediktif\n8 model dan evaluasi\nPermutation importance', 'process')
causal = box(ax, 5.1, 5.35, 4.1, 1.3,
             '4B. Jalur kausal\nGraf berbatas tier + bootstrap\nDML 4 fold dan sensitivitas', 'causal')
arrow(ax, prep['bottom'], pred['top'])
arrow(ax, prep['bottom'], causal['top'])
ext_model = box(ax, 9.65, 6.0, 3.85, 2.2,
                'Jalur eksternal\n7 konstruk\nSplit 75% / 25%\nLatih Logistic Regression\nEvaluasi prediktif', 'process')
down(ax, external, ext_model)
decision = box(ax, .5, 3.4, 8.7, 1.3,
               '5. Bandingkan peringkat prediktif dan kausal\nSimulasikan empat kebijakan dengan anggaran 20%', 'output')
arrow(ax, pred['bottom'], decision['top'])
arrow(ax, causal['bottom'], decision['top'])
assets = box(ax, .5, 1.75, 13, 1.05,
             '6. Artefak tersimpan: model .joblib, tabel CSV, laporan JSON, dan gambar\nDashboard Streamlit membaca artefak untuk enam halaman', 'output')
arrow(ax, decision['bottom'], assets['top'])
arrow(ax, ext_model['bottom'], (ext_model['bottom'][0], assets['top'][1]))
ax.text(.5, .65,
        'Catatan: jalur eksternal melatih model baru; efek kausal tetap bergantung pada asumsi.\n'
        'Nilai kebijakan adalah simulasi. CSV kebijakan saat ini perlu diselaraskan dengan naskah.',
        color='#9a3412', fontsize=11, va='center', linespacing=1.6)
save(fig, 'alur_riset_cfv_dss_id')

fig, ax = canvas('Alur analisis kausal dan pemeriksaan hasil',
                 'Pisahkan struktur graf, estimasi efek, dan diagnostik ketidakpastian.', 13)
nodes = []
steps = [
    (10.55, '1. Matriks kausal seluruh kohor\n13 konstruk hasil preprocessing + TARGET', 'data'),
    (8.9, '2. Discovery berbatas domain\n|Spearman| ≥ 0,04 | arah mengikuti tier | 50 bootstrap', 'causal'),
    (7.1, '3. Graf konsensus dan adjustment set\nRetensi ≥ 60% | singkirkan exposure, outcome, dan descendant', 'causal'),
    (5.3, '4. DML dengan cross-fitting 4 fold\nStandarkan exposure ke 1 SD; prediksi Y dan D dari W', 'process'),
    (3.5, '5. Residualisasi dan estimasi koefisien\nEfek per SD + interval 95% + placebo residual', 'output'),
]
for y, label, kind in steps:
    nodes.append(box(ax, .5, y, 8.1, 1.0 if kind == 'data' else 1.2, label, kind))
for a, b in zip(nodes, nodes[1:]):
    down(ax, a, b)
checks = box(ax, 9.0, 4.3, 4.5, 3.2,
             '6. Pemeriksaan sensitivitas\n\nAmbang graf: 60 / 70 / 80%\nOverlap exposure\nTrimming persentil 1–99\nConfounding tidak terukur', 'note')
arrow(ax, nodes[-1]['right'], checks['left'])
out = box(ax, .5, 1.2, 13, 1.5,
          '7. Gabungkan efek dengan importance prediktif\nTelaah perbedaan peringkat dan kandidat lever\nSajikan graf, tabel efek, serta diagnostik pada Causal Diagnosis', 'output')
arrow(ax, nodes[-1]['bottom'], out['top'])
arrow(ax, checks['bottom'], out['top'])
ax.text(.5, .55, 'AAP: tanda berlawanan dengan ekspektasi dan overlap terbatas; jangan dibaca sebagai manfaat intervensi.',
        fontsize=10.8, color='#9a3412')
save(fig, 'alur_analisis_kausal_id')

fig, ax = canvas('Alur penilaian pemohon di Streamlit',
                 'Model dan koefisien sudah tersedia; halaman ini tidak melatih ulang model.', 13)
inputs = box(ax, .5, 10.25, 8.3, 1.25,
             '1. Buka Individual Assessment\nMasukkan 13 nilai konstruk sesuai rumus fitur', 'data')
assets = box(ax, 9.25, 10.25, 4.25, 1.25,
             'Artefak tersimpan\nModel bundle + tabel efek', 'data')
button = box(ax, .5, 8.65, 8.3, 1.2,
             '2. Klik “Hitung risiko dan prioritas”\nSusun input sesuai urutan feature_names', 'process')
down(ax, inputs, button)
prob = box(ax, .5, 5.65, 6.1, 2.15,
           '3A. Hitung probabilitas risiko\n\nPreprocessor tersimpan mentransformasi input\nModel menghitung predict_proba kelas 1', 'process')
priority = box(ax, 7.15, 5.65, 6.35, 2.15,
               '3B. Hitung skor kandidat lever\n\n|koefisien DML| × kelayakan × max(0, input)\nKoefisien rata-rata sama bagi setiap pemohon', 'causal')
arrow(ax, button['bottom'], prob['top'])
arrow(ax, button['bottom'], priority['top'])
arrow(ax, assets['bottom'], button['right'])
level = box(ax, .5, 3.25, 6.1, 1.65,
            '4A. Tampilkan kategori risiko\nRendah: p < 0,12\nSedang: 0,12 ≤ p < 0,30 | Tinggi: p ≥ 0,30', 'output')
rank = box(ax, 7.15, 3.25, 6.35, 1.65,
           '4B. Urutkan kandidat berdasarkan skor\nTampilkan exposure, efek, interval 95%,\nactionability, dan skor prioritas', 'output')
down(ax, prob, level)
down(ax, priority, rank)
review = box(ax, .5, 1.15, 13, 1.25,
             '5. Pengguna menelaah risiko dan bukti kandidat tindakan\nSkor prioritas merupakan urutan kandidat; dampak intervensi aktual belum diuji', 'note')
arrow(ax, level['bottom'], review['top'])
arrow(ax, rank['bottom'], review['top'])
ax.text(.5, .5,
        'Dua cabang dihitung terpisah: skor prioritas individual pada kode saat ini tidak dikalikan probabilitas risiko.',
        fontsize=10.8, color='#475569')
save(fig, 'alur_penilaian_streamlit_id')
