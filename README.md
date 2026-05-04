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
  run_annotation.py   pipeline de extração

tools/
  scripts auxiliares para debug e exploração
```

---

## 🚀 Pipeline atual

### ✔ Extração de vistas (implementado)

* Processamento em lote (~300 imagens em ~47s)
* Estratégia focada no nadir:

  * pitch ≈ -85°
  * yaw ∈ [0°, 120°, 240°]
* Geração de múltiplas vistas por imagem

Cada vista inclui:

* Imagem perspectiva
* Metadados (yaw, pitch, FOV, tamanho)

---

## 🛠 Instalação

```bash
pip install -e .
```

---

## 🧪 Execução

```bash
python -m scripts.run_annotation
```

---

## 📊 Desempenho

* ~6 imagens/segundo (extração de vistas)
* ~900 vistas geradas para ~300 imagens

---

## 📌 Próximos passos

* Segmentação automática com modelos fundacionais
* Reprojeção e fusão de máscaras
* Refinamento com técnicas clássicas
* Treinamento de modelo especialista (YOLO-seg)
* Exportação ONNX + benchmark CPU

```
```
