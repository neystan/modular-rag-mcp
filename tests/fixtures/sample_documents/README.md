# 样例文档

该目录存放摄取流程和集成测试使用的最小 PDF 样本：

- `simple.pdf`：纯文本单页 PDF，用于基础回归。
- `with_images.pdf`：单页带一张嵌入图片的 PDF，用于 Loader 图片抽取验证。
- `complex_technical_doc.pdf`：多页技术文档，包含 8 个章节、5 个表格和 3 张嵌入图片，用于 C14 pipeline 流程验证。

如需重新生成这些文件，执行：

```bash
uv run python tests/fixtures/sample_documents/generate_pdfs.py
```
