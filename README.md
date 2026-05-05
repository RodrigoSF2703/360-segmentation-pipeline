# 360 Segmentation Pipeline

Pipeline de visão computacional para **segmentação de motocicleta e condutor em imagens 360° (equiretangulares)**, com geração automática de anotações, reprojeção geométrica e treinamento de modelo especialista.

---

## 🎯 Objetivo

Este projeto implementa um pipeline completo para:

* Extração de vistas perspectiva a partir de imagens 360°
* Segmentação automática (zero-shot)
* Reprojeção para o espaço equiretangular
* Refinamento de máscaras
* Treinamento de modelo especialista
* Exportação para ONNX e avaliação em CPU

---

## 🧠 Observações do Dataset

* Imagens no formato **equiretangular (2:1)**
* Resolução típica: **5760 x 2880**
* Os objetos de interesse (motocicleta e condutor) estão localizados predominantemente na região inferior (**nadir/polo sul**)

---

## ⚙️ Estratégia

Ao invés de processar toda a esfera, o pipeline utiliza uma abordagem otimizada:

* Extração de vistas perspectiva focadas no nadir
* Redução de ruído e custo computacional
* Melhor cobertura da região relevante para segmentação

---

## 📦 Estrutura do Projeto

```
data/
  raw/                imagens originais
  intermediate/       vistas e máscaras intermediárias
  processed/          dataset final

src/
  extraction/         conversão 360° → perspectiva
  segmentation/       modelos zero-shot
  reprojection/       volta para equiretangular
  refinement/         pós-processamento
  training/           treino do modelo
  evaluation/         métricas e benchmark

scripts/
  run_annotation.py     extração de vistas
  run_segmentation.py   segmentação + reprojeção + fusão

tools/
  scripts auxiliares para debug e exploração
```

---

## 🚀 Pipeline atual

### ✔ Extração de vistas

Três modos disponíveis via [configs/pipeline.yaml](configs/pipeline.yaml):

* `single`: uma vista (yaw + pitch escalares).
* `nadir_multi`: N yaws com pitch fixo (default: pitch=-85, yaws=[0, 120, 240]).
  Foi o modo usado para gerar as máscaras do nadir, com overlap entre as três
  vistas alimentando a fusão.
* `grid`: varredura por incrementos de yaw e pitch (`yaw_step`, `pitch_step`).
  Os polos exatos são pulados, porque a perspectiva degenera lá.

Cada vista inclui imagem perspectiva + metadados (yaw, pitch, FOV, tamanho).
Processamento em lote: ~300 imagens em ~47s no modo `nadir_multi`.

### ✔ Segmentação zero-shot (LangSAM)

* Prompt textual (ex: `"person on a motorcycle"`)
* Inferência por vista plana, onde a distorção é menor do que no equiretangular cru

### ✔ Reprojeção e fusão equiretangular

* Inverse mapping vetorizado com `cv2.remap`
* Fusão por voto com threshold configurável

---

## 🧭 Reprojeção e a "pegadinha" geométrica

A reprojeção das máscaras planas de volta ao equiretangular é a parte mais
delicada do pipeline. A área coberta por um pixel da vista plana cresce
conforme se aproxima do polo, e como o nadir é onde a moto e o condutor
estão, é lá que a distorção mais aparece.

Forward mapping (varrer pixels da vista plana e escrever no equiretangular)
não funciona bem aqui: perto do polo um pixel plano cobre uma região larga
do equiretangular, e a máscara final acaba pontilhada, com buracos.

O `Reprojector` faz o caminho inverso: varre os pixels do equiretangular e,
para cada um, calcula de qual pixel da vista plana ele veio. A amostragem
fica em uma única chamada de `cv2.remap` por vista. As etapas são:

1. Converter cada pixel `(u, v)` do equiretangular em um vetor unitário 3D
   na esfera (longitude/latitude para XYZ).
2. Aplicar a transposta da rotação da câmera (`R.T`) para obter o vetor no
   referencial da câmera.
3. Descartar pontos com `z <= 0` (atrás da câmera).
4. Projetar em `(x_plane, y_plane)` usando o modelo pinhole.
5. Marcar pixels fora dos limites da imagem plana como inválidos
   (coordenada `-1`).
6. `cv2.remap` com `INTER_NEAREST` e `BORDER_CONSTANT=0` faz a amostragem.

A convenção de eixos segue o `py360convert` usado na extração — assim o
yaw/pitch da vista plana significa a mesma coisa nas duas etapas (detalhes
na docstring de [Reprojector](src/reprojection/reprojector.py)).

### Fusão de máscaras com overlap

A fusão é feita por voto por pixel (`fuse_masks`) com threshold configurável:

* `threshold=1`: união (equivale a `np.maximum`).
* `threshold>=2`: exige overlap entre vistas, fica mais robusto a falsos
  positivos isolados, mas pode perder objeto quando o overlap não existe
  (frequente no nadir, com 3 yaws espaçados 120°).

Padrão atual: `threshold=1`. A limpeza de falsos positivos remanescentes fica
para a etapa de refinamento clássico.

---

## 🛠 Instalação

```bash
pip install -e .
```

---

## 🧪 Execução

Extração de vistas:

```bash
python -m scripts.run_annotation
```

Segmentação zero-shot + reprojeção + fusão:

```bash
python -m scripts.run_segmentation
```

---

## 📊 Desempenho

* Extração: ~6 imagens/s, ~900 vistas para ~300 imagens
* Reprojeção: vetorizada com `cv2.remap`, processa as 3 vistas de uma imagem
  5760x2880 em poucos segundos por imagem

---

## 📌 Próximos passos

* Refinamento com técnicas clássicas (morfológicas, GrabCut, limiarização)
* Treinamento de modelo especialista (YOLO-seg)
* Exportação ONNX + benchmark CPU
* Imagens de exemplo (original 360° vs. máscara final)
* Métricas finais (IoU, mAP, FPS em CPU)
