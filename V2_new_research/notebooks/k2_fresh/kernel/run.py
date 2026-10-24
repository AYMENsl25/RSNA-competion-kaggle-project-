"""K1 Kaggle runner: two fine-tunes of nartaa's public Raptor-384 in parallel, one per T4.

1. find inputs, check the parent checkpoint sha256
2. consensus soft labels from five public LLM report-label tables (gold-58 excluded from training)
3. parity gate: rebuild 3 training studies from DICOM with the reader embedded in the core and compare
   with the tensor rows (centre 352 px); any mismatch stops the run
4. arm A on GPU 0, arm B on GPU 1 (train_ft.py, embedded below); progress in /kaggle/working/status.json
"""
import os, sys, json, time, glob, base64, hashlib, subprocess, shutil
import numpy as np, pandas as pd

T0 = time.time()
W = "/kaggle/working"
HARD_STOP_H = 11.4
BUDGET_H = float(os.environ.get("K1_BUDGET_H", "9.0"))
TRAIN_B64 = "IiIiSzE6IGZpbmUtdHVuZSBuYXJ0YWEncyBwdWJsaWMgUmFwdG9yLTM4NCBDb0F0TmV0IChMQiAwLjk0OSkgb24gbG9ubmllcWluJ3MgOTYtc2xpY2UgdGVuc29ycy4KClRoZSB0ZW5zb3JzIHdlcmUgYnVpbHQgYnkgdGhlIHNhbWUgZW1iZWRkZWQgRElDT00gcmVhZGVyIHRoZSAwLjk1OCBub3RlYm9vayB1c2VzIGF0IHRlc3QgdGltZSAoZGVjb2RlZApzb3VyY2Ugc2hhIDVhNTliNmY4Li4uKSwgc3RvcmVkIGFzIHRoZSBjZW50cmUgMzUyIHB4IG9mIHRoZSAzODQgcHggLyAxNDAgbW0gdm9sdW1lLCBzbyB0aGUgMzIwIHB4IGNlbnRyZQpjcm9wIGhlcmUgaXMgdGhlIGluZmVyZW5jZSBpbnB1dC4gT25lIGFybSBwZXIgR1BVOgoKICBBICBhcmNoaXRlY3R1cmUgdW5jaGFuZ2VkLiBUYXJnZXRzID0gYWxwaGEgKiBjb25zZW5zdXMgb2YgNSBwdWJsaWMgTExNIHJlcG9ydC1sYWJlbCB0YWJsZXMKICAgICArICgxIC0gYWxwaGEpICogZnJvemVuIHBhcmVudCBwcmVkaWN0aW9uIG9uIHRoZSBzYW1lIGF1Z21lbnRlZCB3aW5kb3dzIChrZWVwcyB3aGF0IHRoZSBwYXJlbnQKICAgICBsZWFybmVkIGZyb20gT0FJKS4gTWlycm9yIC8gY3JvcC1zaGlmdCAvIGdhaW4gLyBnYW1tYSBhdWdtZW50YXRpb24uCiAgQiAgc2FtZSB0YXJnZXRzLCBkdWFsIHZpZXc6IHRoZSBnbG9iYWwgMzIwIGNyb3AgcGx1cyBhIGNlbnRyZSB6b29tIChmcmFjIG9mIHRoZSAzNTIgcHgsIHJlc2l6ZWQgdG8KICAgICAzMjApIG9mIHRoZSBzYW1lIHdpbmRvd3MsIHplcm8taW5pdGlhbGlzZWQgKHZpZXcsIHNsb3QpIGVtYmVkZGluZ3MgYW5kIGEgbGVhcm5lZCBhdHRlbnRpb24gb2Zmc2V0CiAgICAgZm9yIHpvb20gd2luZG93cyAoc3RhcnRzIGF0IC0yLCBzbyB0aGUgc3RhcnQgcG9pbnQgaXMgY2xvc2UgdG8gdGhlIHBhcmVudCkuCgpHb2xkLTU4IGlzIG5ldmVyIHRyYWluZWQgb247IGl0cyBBVUMgaXMgcHJpbnRlZCBwZXIgZXBvY2ggZm9yIGluZm9ybWF0aW9uIG9ubHkgKHRoZSBwYXJlbnQncyBTV0EgZXBvY2hzCndlcmUgcGlja2VkIG9uIGdvbGQtNTgsIHNvIHRoZSBwYXJlbnQncyBvd24gZ29sZCBzY29yZSBpcyBvcHRpbWlzdGljKS4gVGhlIHByZWRlY2xhcmVkIHJlc3VsdCBpcyB0aGUKbGFzdCBlcG9jaDsgdGhlIHB1YmxpYyBMQiBkZWNpZGVzLgoiIiIKaW1wb3J0IG9zLCBzeXMsIGpzb24sIHRpbWUsIG1hdGgsIGFyZ3BhcnNlLCBjb3B5CmltcG9ydCBudW1weSBhcyBucCwgcGFuZGFzIGFzIHBkCmltcG9ydCB0b3JjaCwgdG9yY2gubm4gYXMgbm4sIHRvcmNoLm5uLmZ1bmN0aW9uYWwgYXMgRgpmcm9tIHRvcmNoLnV0aWxzLmNoZWNrcG9pbnQgaW1wb3J0IGNoZWNrcG9pbnQKZnJvbSBza2xlYXJuLm1ldHJpY3MgaW1wb3J0IHJvY19hdWNfc2NvcmUKCmFwID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQphcC5hZGRfYXJndW1lbnQoIi0tYXJtIiwgY2hvaWNlcz1bIkEiLCAiQiJdLCByZXF1aXJlZD1UcnVlKQphcC5hZGRfYXJndW1lbnQoIi0tY29yZSIsIHJlcXVpcmVkPVRydWUsIGhlbHA9ImRpcmVjdG9yeSBob2xkaW5nIHpvb19yYXB0b3JfY29yZS5weSIpCmFwLmFkZF9hcmd1bWVudCgiLS1wYXJlbnQiLCBkZWZhdWx0PSIiLCBoZWxwPSJwYXJlbnQgY2hlY2twb2ludDsgZW1wdHkgPSByYW5kb20gaW5pdCAoc21va2Ugb25seSkiKQphcC5hZGRfYXJndW1lbnQoIi0tYXJjaCIsIGRlZmF1bHQ9ImNvYXRuZXRfcm1scF8yX3J3XzM4NC5zd19pbjEya19mdF9pbjFrIikKYXAuYWRkX2FyZ3VtZW50KCItLWRhdGFfcm9vdCIsIGRlZmF1bHQ9Ii9rYWdnbGUvaW5wdXQiKQphcC5hZGRfYXJndW1lbnQoIi0tbGFiZWxzX2pzb24iLCByZXF1aXJlZD1UcnVlLCBoZWxwPSJqc29uIHdyaXR0ZW4gYnkgcnVuLnB5OiB7dWlkOiBbMTIgZmxvYXRzIG9yIG51bGxdfSwgZ29sZCBsaXN0IikKYXAuYWRkX2FyZ3VtZW50KCItLW91dCIsIHJlcXVpcmVkPVRydWUpCmFwLmFkZF9hcmd1bWVudCgiLS1idWRnZXRfaCIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9OC4wKQphcC5hZGRfYXJndW1lbnQoIi0tZW1heCIsIHR5cGU9aW50LCBkZWZhdWx0PTYpCmFwLmFkZF9hcmd1bWVudCgiLS1UIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MTIpCmFwLmFkZF9hcmd1bWVudCgiLS1hY2N1bSIsIHR5cGU9aW50LCBkZWZhdWx0PTE2KQphcC5hZGRfYXJndW1lbnQoIi0tY2h1bmsiLCB0eXBlPWludCwgZGVmYXVsdD00KQphcC5hZGRfYXJndW1lbnQoIi0tYmJfbHIiLCB0eXBlPWZsb2F0LCBkZWZhdWx0PTEuNWUtNSkKYXAuYWRkX2FyZ3VtZW50KCItLWhlYWRfbHIiLCB0eXBlPWZsb2F0LCBkZWZhdWx0PTNlLTQpCmFwLmFkZF9hcmd1bWVudCgiLS1uZXdfbHIiLCB0eXBlPWZsb2F0LCBkZWZhdWx0PTFlLTMpCmFwLmFkZF9hcmd1bWVudCgiLS13ZCIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MC4wNSkKYXAuYWRkX2FyZ3VtZW50KCItLWFscGhhIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0wLjUpCmFwLmFkZF9hcmd1bWVudCgiLS16b29tIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0wLjYpCmFwLmFkZF9hcmd1bWVudCgiLS1rX2V2YWwiLCB0eXBlPWludCwgZGVmYXVsdD05NCkKYXAuYWRkX2FyZ3VtZW50KCItLXdvcmtlcnMiLCB0eXBlPWludCwgZGVmYXVsdD0yKQphcC5hZGRfYXJndW1lbnQoIi0tbGltaXQiLCB0eXBlPWludCwgZGVmYXVsdD0wLCBoZWxwPSJzbW9rZTogdHJhaW4gb24gTiBzdHVkaWVzIikKYXAuYWRkX2FyZ3VtZW50KCItLWdvbGRfbGltaXQiLCB0eXBlPWludCwgZGVmYXVsdD0wKQphcC5hZGRfYXJndW1lbnQoIi0tc2VlZCIsIHR5cGU9aW50LCBkZWZhdWx0PTcpCmFwLmFkZF9hcmd1bWVudCgiLS1wcmV0cmFpbmVkIiwgYWN0aW9uPSJzdG9yZV90cnVlIiwgaGVscD0idGltbSBJbWFnZU5ldCB3ZWlnaHRzIChLMiBmcmVzaCBtZW1iZXJzOyBuZWVkcyBpbnRlcm5ldCkiKQphcC5hZGRfYXJndW1lbnQoIi0tdmlldyIsIGNob2ljZXM9WyJnbG9iYWwiLCAiem9vbSJdLCBkZWZhdWx0PSJnbG9iYWwiLCBoZWxwPSJhcm0gQSBpbnB1dCB2aWV3IikKYSA9IGFwLnBhcnNlX2FyZ3MoKQoKc3lzLnBhdGguaW5zZXJ0KDAsIGEuY29yZSkKaW1wb3J0IHpvb19yYXB0b3JfY29yZSBhcyBDICAjIG5vcWE6IEU0MDIKCnRvcmNoLm1hbnVhbF9zZWVkKGEuc2VlZCk7IG5wLnJhbmRvbS5zZWVkKGEuc2VlZCkKZGV2ID0gdG9yY2guZGV2aWNlKCJjdWRhIiBpZiB0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpIGVsc2UgImNwdSIpCm9zLm1ha2VkaXJzKGEub3V0LCBleGlzdF9vaz1UcnVlKQpUMCA9IHRpbWUudGltZSgpCkxBQiA9IEMuTEFCCkQgPSBDLk1BWFMKCgpkZWYgbG9nKCoqa3cpOgogICAga3dbInQiXSA9IHJvdW5kKHRpbWUudGltZSgpIC0gVDAsIDEpCiAgICBsaW5lID0ganNvbi5kdW1wcyhrdykKICAgIHByaW50KGxpbmUsIGZsdXNoPVRydWUpCiAgICB3aXRoIG9wZW4ob3MucGF0aC5qb2luKGEub3V0LCAibG9nLmpzb25sIiksICJhIikgYXMgZjoKICAgICAgICBmLndyaXRlKGxpbmUgKyAiXG4iKQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBkYXRhCmRlZiBmaW5kKG5hbWUpOgogICAgZm9yIGQsIGRpcnMsIGZzIGluIG9zLndhbGsoYS5kYXRhX3Jvb3QpOgogICAgICAgIGRpcnNbOl0gPSBbeCBmb3IgeCBpbiBkaXJzIGlmICJhYm5vcm1hbGl0eS1kZXRlY3Rpb24iIG5vdCBpbiB4IGFuZCB4ICE9ICJjb21wZXRpdGlvbnMiXQogICAgICAgIGlmIG5hbWUgaW4gZnM6CiAgICAgICAgICAgIHJldHVybiBvcy5wYXRoLmpvaW4oZCwgbmFtZSkKICAgIHJhaXNlIEZpbGVOb3RGb3VuZEVycm9yKG5hbWUpCgoKVk9MUywgTUFTS1MsIExPQyA9IFtdLCBbXSwge30KZm9yIHMgaW4gcmFuZ2UoNCk6CiAgICBWT0xTLmFwcGVuZChucC5sb2FkKGZpbmQoZiJ2b2xfc3tzfS51OC5ucHkiKSwgbW1hcF9tb2RlPSJyIikpCiAgICBNQVNLUy5hcHBlbmQobnAubG9hZChmaW5kKGYibWFza19ze3N9LnU4Lm5weSIpKSkKICAgIGlkeCA9IHBkLnJlYWRfY3N2KGZpbmQoZiJpbmRleF9ze3N9LmNzdiIpKQogICAgZm9yIHVpZCwgcm93LCBkb25lIGluIHppcChpZHguU3R1ZHlJbnN0YW5jZVVJRC5hc3R5cGUoc3RyKSwgaWR4LnJvdywgaWR4LmRvbmUpOgogICAgICAgIGlmIGRvbmU6CiAgICAgICAgICAgIExPQ1t1aWRdID0gKHMsIGludChyb3cpKQpTID0gVk9MU1swXS5zaGFwZVstMV0KYXNzZXJ0IFZPTFNbMF0uc2hhcGVbMV0gPT0gRCBhbmQgUyA9PSAzNTIsIFZPTFNbMF0uc2hhcGUKCmxhYiA9IGpzb24ubG9hZChvcGVuKGEubGFiZWxzX2pzb24pKQpnb2xkID0gW3UgZm9yIHUgaW4gbGFiWyJnb2xkIl0gaWYgdSBpbiBMT0NdCmdvbGRfeSA9IHt1OiBucC5hcnJheShsYWJbImdvbGRfeSJdW3VdLCBucC5mbG9hdDMyKSBmb3IgdSBpbiBnb2xkfQp0cmFpbiA9IFt1IGZvciB1LCB2IGluIGxhYlsieSJdLml0ZW1zKCkgaWYgdSBpbiBMT0MgYW5kIHUgbm90IGluIHNldChsYWJbImdvbGQiXSldClkgPSB7dTogbnAuYXJyYXkoW25wLm5hbiBpZiB4IGlzIE5vbmUgZWxzZSB4IGZvciB4IGluIGxhYlsieSJdW3VdXSwgbnAuZmxvYXQzMikgZm9yIHUgaW4gdHJhaW59CmlmIGEubGltaXQ6CiAgICB0cmFpbiA9IHRyYWluWzphLmxpbWl0XQppZiBhLmdvbGRfbGltaXQ6CiAgICBnb2xkID0gZ29sZFs6YS5nb2xkX2xpbWl0XQpwcmV2ID0gbnAuY2xpcChucC5uYW5tZWFuKG5wLnN0YWNrKFtZW3VdIGZvciB1IGluIHRyYWluXSksIDApLCAwLjAzLCAwLjcpClBXID0gdG9yY2gudGVuc29yKG5wLmNsaXAoKDEgLSBwcmV2KSAvIHByZXYsIDEsIDEwKSwgZHR5cGU9dG9yY2guZmxvYXQzMiwgZGV2aWNlPWRldikKbG9nKGV2PSJkYXRhIiwgYXJtPWEuYXJtLCB0cmFpbj1sZW4odHJhaW4pLCBnb2xkPWxlbihnb2xkKSwgc2hhcmRzPVt2LnNoYXBlWzBdIGZvciB2IGluIFZPTFNdLCBwdz1QVy50b2xpc3QoKSkKCgpjbGFzcyBCYWdzKHRvcmNoLnV0aWxzLmRhdGEuRGF0YXNldCk6CiAgICBkZWYgX19pbml0X18oc2VsZiwgdWlkcyk6CiAgICAgICAgc2VsZi51aWRzID0gdWlkcwoKICAgIGRlZiBfX2xlbl9fKHNlbGYpOgogICAgICAgIHJldHVybiBsZW4oc2VsZi51aWRzKQoKICAgIGRlZiBfX2dldGl0ZW1fXyhzZWxmLCBpKToKICAgICAgICB1ID0gc2VsZi51aWRzW2ldCiAgICAgICAgcywgciA9IExPQ1t1XQogICAgICAgIHJuZyA9IG5wLnJhbmRvbS5kZWZhdWx0X3JuZygodG9yY2guaW5pdGlhbF9zZWVkKCkgKyBpKSAlIDIqKjMyKQogICAgICAgIGNzID0gQy50cmFpbl9jZW50ZXJzKE1BU0tTW3NdW3JdLCBELCBhLlQsIHJuZykKICAgICAgICB0cmkgPSBucC5hc2NvbnRpZ3VvdXNhcnJheShWT0xTW3NdW3JdW25wLnN0YWNrKFtjcyAtIDEsIGNzLCBjcyArIDFdLCAxKV0pICAjIFtULDMsUyxTXSB1aW50OAogICAgICAgIHJldHVybiB0b3JjaC5mcm9tX251bXB5KHRyaSksIHRvcmNoLmZyb21fbnVtcHkoY3MpLCB0b3JjaC5mcm9tX251bXB5KFlbdV0pCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIG1vZGVsCmRlZiBsb2FkX3N0YXRlKHBhdGgpOgogICAgY2sgPSB0b3JjaC5sb2FkKHBhdGgsIG1hcF9sb2NhdGlvbj0iY3B1Iiwgd2VpZ2h0c19vbmx5PUZhbHNlKQogICAgcmV0dXJuIGNrWyJtb2RlbCJdCgoKcGFyZW50X3NkID0gbG9hZF9zdGF0ZShhLnBhcmVudCkgaWYgYS5wYXJlbnQgZWxzZSBOb25lCmlmIGEuYXJjaC5zdGFydHN3aXRoKCgiY29hdG5ldCIsICJtYXh2aXQiLCAiY29udm5leHQiKSk6CiAgICBzdHVkZW50ID0gQy5idWlsZF9tb2RlbChhcmNoPWEuYXJjaCwgc3RhdGU9cGFyZW50X3NkLCBwcmV0cmFpbmVkPWEucHJldHJhaW5lZCBhbmQgcGFyZW50X3NkIGlzIE5vbmUpCmVsc2U6ICAjIEsyOiBDTk4gZmFtaWxpZXMgd2hvc2UgdGltbSBjb25zdHJ1Y3RvciB0YWtlcyBubyBpbWdfc2l6ZSAoRWZmaWNpZW50TmV0VjIgZXRjLikKICAgIGltcG9ydCB0aW1tCiAgICBfYmIgPSB0aW1tLmNyZWF0ZV9tb2RlbChhLmFyY2gsIHByZXRyYWluZWQ9YS5wcmV0cmFpbmVkLCBudW1fY2xhc3Nlcz0wLCBnbG9iYWxfcG9vbD0iYXZnIikKICAgIHN0dWRlbnQgPSBDLlJhcHRvckNsYXNzaWZpZXIoX2JiLCBGX2RpbT1fYmIubnVtX2ZlYXR1cmVzKQogICAgaWYgcGFyZW50X3NkIGlzIG5vdCBOb25lOgogICAgICAgIHN0dWRlbnQubG9hZF9zdGF0ZV9kaWN0KHtrOiB2LmZsb2F0KCkgZm9yIGssIHYgaW4gcGFyZW50X3NkLml0ZW1zKCl9LCBzdHJpY3Q9VHJ1ZSkKQk4gPSBbbSBmb3IgbSBpbiBzdHVkZW50Lm1vZHVsZXMoKSBpZiBpc2luc3RhbmNlKG0sIG5uLm1vZHVsZXMuYmF0Y2hub3JtLl9CYXRjaE5vcm0pXQpsb2coZXY9Im1vZGVsIiwgYXJjaD1hLmFyY2gsIHByZXRyYWluZWQ9YS5wcmV0cmFpbmVkLCBibl9sYXllcnM9bGVuKEJOKSwgdmlldz1hLnZpZXcsCiAgICBwYXJhbXNfbT1yb3VuZChzdW0ocC5udW1lbCgpIGZvciBwIGluIHN0dWRlbnQucGFyYW1ldGVycygpKSAvIDFlNiwgMSkpCnRlYWNoZXIgPSBjb3B5LmRlZXBjb3B5KHN0dWRlbnQpLmhhbGYoKS5ldmFsKCkudG8oZGV2KSBpZiBwYXJlbnRfc2QgaXMgbm90IE5vbmUgZWxzZSBOb25lCmlmIHRlYWNoZXIgaXMgbm90IE5vbmU6CiAgICBmb3IgcCBpbiB0ZWFjaGVyLnBhcmFtZXRlcnMoKToKICAgICAgICBwLnJlcXVpcmVzX2dyYWRfKEZhbHNlKQpTTE9UID0gdG9yY2guZnJvbV9udW1weShDLl9zbG90X2luZGV4KEQpWzBdKS5sb25nKCkudG8oZGV2KSAgIyBzbGljZSAtPiBzbG90IDAuLjQKCgpjbGFzcyBEdWFsVmlldyhubi5Nb2R1bGUpOgogICAgIiIiUmFwdG9yIGhlYWQgb3ZlciBnbG9iYWwgKyB6b29tIHdpbmRvd3M7IHplcm8gKHZpZXcsIHNsb3QpIGVtYmVkZGluZ3Mga2VlcCB0aGUgcGFyZW50IGZ1bmN0aW9uIG9uIHRoZQogICAgZ2xvYmFsIHZpZXcgYXQgc3RlcCAwLCB6b29tIHdpbmRvd3Mgc3RhcnQgd2l0aCBhbiBhdHRlbnRpb24gb2Zmc2V0IG9mIC0yLiIiIgoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBiYXNlKToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKCkKICAgICAgICBzZWxmLmJhc2UgPSBiYXNlCiAgICAgICAgZmQgPSBiYXNlLmNsc1cuc2hhcGVbMV0KICAgICAgICBzZWxmLmVtYiA9IG5uLlBhcmFtZXRlcih0b3JjaC56ZXJvcygxMCwgZmQpKQogICAgICAgIHNlbGYuemJpYXMgPSBubi5QYXJhbWV0ZXIodG9yY2guZnVsbCgoYmFzZS5uLCksIC0yLjApKQoKICAgIGRlZiBwb29sKHNlbGYsIGZnLCBmeiwgc2xvdCk6CiAgICAgICAgZiA9IHRvcmNoLmNhdChbZmcgKyBzZWxmLmVtYltzbG90XSwgZnogKyBzZWxmLmVtYltzbG90ICsgNV1dLCAxKQogICAgICAgIGggPSBzZWxmLmJhc2Uubm9ybShmKQogICAgICAgIGF0dCA9IHNlbGYuYmFzZS5hdHQoaCkKICAgICAgICBrID0gZmcuc2hhcGVbMV0KICAgICAgICBhdHQgPSB0b3JjaC5jYXQoW2F0dFs6LCA6a10sIGF0dFs6LCBrOl0gKyBzZWxmLnpiaWFzXSwgMSkuZmxvYXQoKS5zb2Z0bWF4KDEpLnRvKGguZHR5cGUpCiAgICAgICAgcG9vbGVkID0gdG9yY2guZWluc3VtKCJia24sYmtmLT5ibmYiLCBhdHQsIGgpCiAgICAgICAgcmV0dXJuIChwb29sZWQgKiBzZWxmLmJhc2UuY2xzVykuc3VtKC0xKSArIHNlbGYuYmFzZS5jbHNiCgoKbW9kZWwgPSAoRHVhbFZpZXcoc3R1ZGVudCkgaWYgYS5hcm0gPT0gIkIiIGVsc2Ugc3R1ZGVudCkudG8oZGV2KQoKCmRlZiBlbmNvZGUobSwgeCwgdHJhaW4pOgogICAgYiwgayA9IHguc2hhcGVbOjJdCiAgICBmbGF0ID0geC5mbGF0dGVuKDAsIDEpCiAgICBpZiB0cmFpbjoKICAgICAgICBmID0gdG9yY2guY2F0KFtjaGVja3BvaW50KG0uYmFja2JvbmUsIGMsIHVzZV9yZWVudHJhbnQ9RmFsc2UpIGZvciBjIGluIGZsYXQuc3BsaXQoYS5jaHVuayldKQogICAgZWxzZToKICAgICAgICBmID0gdG9yY2guY2F0KFttLmJhY2tib25lKGMpIGZvciBjIGluIGZsYXQuc3BsaXQoMTYpXSkKICAgIHJldHVybiBmLnZpZXcoYiwgaywgLTEpCgoKZGVmIHpvb21fdmlld3ModHJpLCBjcywgbWlycm9yLCBvZmZzLCBnYWluLCBnYW1tYSk6CiAgICAiIiJDZW50cmUgem9vbSBvZiB0aGUgc2FtZSB1aW50OCB3aW5kb3dzOiBjcm9wIGZyYWMqUyAoc2hpZnRlZCBsaWtlIHRoZSBnbG9iYWwgY3JvcCksIHJlc2l6ZSB0byAzMjAuIiIiCiAgICBiID0gdHJpLnNoYXBlWzBdCiAgICB6ID0gaW50KHJvdW5kKGEuem9vbSAqIFMpKQogICAgcm93cyA9IFtdCiAgICBmb3IgaSBpbiByYW5nZShiKToKICAgICAgICB4ID0gdHJpW2ldCiAgICAgICAgaWYgbWlycm9yIGlzIG5vdCBOb25lIGFuZCBib29sKG1pcnJvcltpXSk6CiAgICAgICAgICAgIHggPSBDLl9taXJyb3JfdHJpKHgsIGNzW2ldLCBEKQogICAgICAgIG95LCBveCA9IChpbnQob2Zmc1tpXVswXSksIGludChvZmZzW2ldWzFdKSkgaWYgb2ZmcyBpcyBub3QgTm9uZSBlbHNlICgwLCAwKQogICAgICAgIHkwID0gbWluKG1heCgoUyAtIHopIC8vIDIgKyBveSwgMCksIFMgLSB6KQogICAgICAgIHgwID0gbWluKG1heCgoUyAtIHopIC8vIDIgKyBveCwgMCksIFMgLSB6KQogICAgICAgIHJvd3MuYXBwZW5kKHhbOiwgOiwgeTA6eTAgKyB6LCB4MDp4MCArIHpdKQogICAgeCA9IHRvcmNoLnN0YWNrKHJvd3MpLmZsb2F0KCkuZGl2XygyNTUuMCkKICAgIGlmIGdhbW1hIGlzIG5vdCBOb25lOgogICAgICAgIHggPSB4LmNsYW1wX21pbigxZS00KS5wb3codG9yY2guYXNfdGVuc29yKGdhbW1hLCBkdHlwZT10b3JjaC5mbG9hdDMyLCBkZXZpY2U9eC5kZXZpY2UpLnZpZXcoYiwgMSwgMSwgMSwgMSkpCiAgICBpZiBnYWluIGlzIG5vdCBOb25lOgogICAgICAgIHggPSAoeCAqIHRvcmNoLmFzX3RlbnNvcihnYWluLCBkdHlwZT10b3JjaC5mbG9hdDMyLCBkZXZpY2U9eC5kZXZpY2UpLnZpZXcoYiwgMSwgMSwgMSwgMSkpLmNsYW1wXygwLCAxKQogICAgayA9IHguc2hhcGVbMV0KICAgIHggPSBGLmludGVycG9sYXRlKHguZmxhdHRlbigwLCAxKSwgc2l6ZT0oQy5JTlBVVF9SRVMsIEMuSU5QVVRfUkVTKSwgbW9kZT0iYmlsaW5lYXIiLCBhbGlnbl9jb3JuZXJzPUZhbHNlKQogICAgeCA9IHgudmlldyhiLCBrLCAzLCBDLklOUFVUX1JFUywgQy5JTlBVVF9SRVMpCiAgICByZXR1cm4gKHggLSBDLl9NRUFOLnRvKHguZGV2aWNlKS52aWV3KDEsIDEsIDMsIDEsIDEpKSAvIEMuX1NURC50byh4LmRldmljZSkudmlldygxLCAxLCAzLCAxLCAxKQoKCmRlZiBmb3J3YXJkKHRyaSwgY3MsIHRyYWluLCBhdWc9Tm9uZSk6CiAgICBtaXJyb3IsIG9mZnMsIGdhaW4sIGdhbW1hID0gYXVnIGlmIGF1ZyBpcyBub3QgTm9uZSBlbHNlIChOb25lLCBOb25lLCBOb25lLCBOb25lKQogICAgaWYgYS5hcm0gPT0gIkEiIGFuZCBhLnZpZXcgPT0gInpvb20iOgogICAgICAgIHh6ID0gem9vbV92aWV3cyh0cmksIGNzLCBtaXJyb3IsIG9mZnMsIGdhaW4sIGdhbW1hKQogICAgICAgIHJldHVybiBzdHVkZW50LmhlYWQoZW5jb2RlKHN0dWRlbnQsIHh6LCB0cmFpbikpLCB4egogICAgeGcgPSBDLmdwdV92aWV3cyh0cmksIGNzLCBELCBDLklOUFVUX1JFUywgbWlycm9yPW1pcnJvciwgb2Zmcz1vZmZzLCBnYWluPWdhaW4sIGdhbW1hPWdhbW1hKQogICAgaWYgYS5hcm0gPT0gIkEiOgogICAgICAgIHJldHVybiBzdHVkZW50LmhlYWQoZW5jb2RlKHN0dWRlbnQsIHhnLCB0cmFpbikpLCB4ZwogICAgeHogPSB6b29tX3ZpZXdzKHRyaSwgY3MsIG1pcnJvciwgb2ZmcywgZ2FpbiwgZ2FtbWEpCiAgICBzbG90ID0gU0xPVFt0b3JjaC5hc190ZW5zb3IoY3MsIGRldmljZT1kZXYpLmxvbmcoKV0KICAgIHJldHVybiBtb2RlbC5wb29sKGVuY29kZShzdHVkZW50LCB4ZywgdHJhaW4pLCBlbmNvZGUoc3R1ZGVudCwgeHosIHRyYWluKSwgc2xvdCksIHhnCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIGdvbGQgbW9uaXRvciAoaW5mZXJlbmNlIHJlY2lwZToga19ldmFsIHdpbmRvd3MsIHBsYWluICsgbWlycm9yKQpAdG9yY2gubm9fZ3JhZCgpCmRlZiBnb2xkX2V2YWwodGFnKToKICAgIGlmIG5vdCBnb2xkOgogICAgICAgIHJldHVybiBOb25lCiAgICBtb2RlbC5ldmFsKCkKICAgIFAsIFlnID0gW10sIFtdCiAgICBmb3IgdSBpbiBnb2xkOgogICAgICAgIHMsIHIgPSBMT0NbdV0KICAgICAgICBjcyA9IEMuZXZhbF9jZW50ZXJzKE1BU0tTW3NdW3JdLCBELCBhLmtfZXZhbCkKICAgICAgICB0cmkgPSB0b3JjaC5mcm9tX251bXB5KG5wLmFzY29udGlndW91c2FycmF5KFZPTFNbc11bcl1bbnAuc3RhY2soW2NzIC0gMSwgY3MsIGNzICsgMV0sIDEpXSkpLnRvKGRldilbTm9uZV0KICAgICAgICBwcyA9IFtdCiAgICAgICAgZm9yIG1pciBpbiAoRmFsc2UsIFRydWUpOgogICAgICAgICAgICB3aXRoIHRvcmNoLmF1dG9jYXN0KGRldi50eXBlLCBkdHlwZT10b3JjaC5mbG9hdDE2LCBlbmFibGVkPWRldi50eXBlID09ICJjdWRhIik6CiAgICAgICAgICAgICAgICBsZywgXyA9IGZvcndhcmQodHJpLCBjc1tOb25lXSwgRmFsc2UsIChbbWlyXSwgTm9uZSwgTm9uZSwgTm9uZSkpCiAgICAgICAgICAgIHBzLmFwcGVuZCh0b3JjaC5zaWdtb2lkKGxnLmZsb2F0KCkpWzBdLmNwdSgpLm51bXB5KCkpCiAgICAgICAgUC5hcHBlbmQobnAubWVhbihwcywgMCkpOyBZZy5hcHBlbmQoZ29sZF95W3VdKQogICAgUCwgWWcgPSBucC5zdGFjayhQKSwgbnAuc3RhY2soWWcpCiAgICBhdWNzID0ge246IGZsb2F0KHJvY19hdWNfc2NvcmUoWWdbOiwgal0sIFBbOiwgal0pKSBmb3IgaiwgbiBpbiBlbnVtZXJhdGUoTEFCKSBpZiBsZW4oc2V0KFlnWzosIGpdKSkgPiAxfQogICAgbWFjcm8gPSBmbG9hdChucC5tZWFuKGxpc3QoYXVjcy52YWx1ZXMoKSkpKSBpZiBhdWNzIGVsc2UgZmxvYXQoIm5hbiIpCiAgICBucC5zYXZlKG9zLnBhdGguam9pbihhLm91dCwgZiJnb2xkX3t0YWd9Lm5weSIpLCBQKQogICAgbG9nKGV2PSJnb2xkIiwgdGFnPXRhZywgbWFjcm89cm91bmQobWFjcm8sIDQpLCBwZXI9e2s6IHJvdW5kKHYsIDMpIGZvciBrLCB2IGluIGF1Y3MuaXRlbXMoKX0pCiAgICBpZiAidHJhaW5fbW9kZSIgaW4gZ2xvYmFscygpOgogICAgICAgIHRyYWluX21vZGUoKQogICAgcmV0dXJuIG1hY3JvCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIG9wdGltaXNlcgpiYiA9IFtwIGZvciBuLCBwIGluIHN0dWRlbnQubmFtZWRfcGFyYW1ldGVycygpIGlmIG4uc3RhcnRzd2l0aCgiYmFja2JvbmUuIildCmhkID0gW3AgZm9yIG4sIHAgaW4gc3R1ZGVudC5uYW1lZF9wYXJhbWV0ZXJzKCkgaWYgbm90IG4uc3RhcnRzd2l0aCgiYmFja2JvbmUuIildCmdyb3VwcyA9IFt7InBhcmFtcyI6IGJiLCAibHIiOiBhLmJiX2xyfSwgeyJwYXJhbXMiOiBoZCwgImxyIjogYS5oZWFkX2xyfV0KaWYgYS5hcm0gPT0gIkIiOgogICAgZ3JvdXBzLmFwcGVuZCh7InBhcmFtcyI6IFttb2RlbC5lbWIsIG1vZGVsLnpiaWFzXSwgImxyIjogYS5uZXdfbHIsICJ3ZWlnaHRfZGVjYXkiOiAwLjB9KQpvcHQgPSB0b3JjaC5vcHRpbS5BZGFtVyhncm91cHMsIHdlaWdodF9kZWNheT1hLndkKQpiYXNlX2xyID0gW2dbImxyIl0gZm9yIGcgaW4gb3B0LnBhcmFtX2dyb3Vwc10Kc2NhbGVyID0gdG9yY2guYW1wLkdyYWRTY2FsZXIoImN1ZGEiLCBlbmFibGVkPWRldi50eXBlID09ICJjdWRhIikKV0FSTSA9IDMwCnBsYW4gPSB7InRvdGFsIjogMTAgKiogOX0KCgpkZWYgc2V0X2xyKHN0ZXApOgogICAgaWYgc3RlcCA8IFdBUk06CiAgICAgICAgbSA9IChzdGVwICsgMSkgLyBXQVJNCiAgICBlbHNlOgogICAgICAgIG0gPSAwLjUgKiAoMSArIG1hdGguY29zKG1hdGgucGkgKiBtaW4oMS4wLCAoc3RlcCAtIFdBUk0pIC8gbWF4KDEsIHBsYW5bInRvdGFsIl0gLSBXQVJNKSkpKQogICAgZm9yIGcsIGIwIGluIHppcChvcHQucGFyYW1fZ3JvdXBzLCBiYXNlX2xyKToKICAgICAgICBnWyJsciJdID0gYjAgKiBtCgoKZGVmIHNhdmUodGFnLCBleHRyYT1Ob25lKToKICAgIHNkID0ge2s6ICh2LmhhbGYoKSBpZiB2LmlzX2Zsb2F0aW5nX3BvaW50KCkgZWxzZSB2KSBmb3IgaywgdiBpbiBtb2RlbC5zdGF0ZV9kaWN0KCkuaXRlbXMoKX0KICAgIGNrID0geyJtb2RlbCI6IHNkLCAiYXJjaCI6IGEuYXJjaCwgInJlcyI6IDM4NCwgImxhYiI6IExBQiwgImFybSI6IGEuYXJtLCAiem9vbSI6IGEuem9vbSwgInRhZyI6IHRhZywKICAgICAgICAgICJwYXJlbnQiOiBvcy5wYXRoLmJhc2VuYW1lKGEucGFyZW50KSwgImFyZ3MiOiB2YXJzKGEpfQogICAgY2sudXBkYXRlKGV4dHJhIG9yIHt9KQogICAgdG9yY2guc2F2ZShjaywgb3MucGF0aC5qb2luKGEub3V0LCBmImsye2Eudmlld31fe3RhZ30ucHQudG1wIikpCiAgICBvcy5yZXBsYWNlKG9zLnBhdGguam9pbihhLm91dCwgZiJrMnthLnZpZXd9X3t0YWd9LnB0LnRtcCIpLCBvcy5wYXRoLmpvaW4oYS5vdXQsIGYiazJ7YS52aWV3fV97dGFnfS5wdCIpKQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSB0cmFpbgpnb2xkX2V2YWwoInBhcmVudCIpCmRsID0gdG9yY2gudXRpbHMuZGF0YS5EYXRhTG9hZGVyKEJhZ3ModHJhaW4pLCBiYXRjaF9zaXplPTEsIHNodWZmbGU9VHJ1ZSwgbnVtX3dvcmtlcnM9YS53b3JrZXJzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwaW5fbWVtb3J5PWRldi50eXBlID09ICJjdWRhIiwgcGVyc2lzdGVudF93b3JrZXJzPWEud29ya2VycyA+IDAsIGRyb3BfbGFzdD1UcnVlKQpybmcgPSBucC5yYW5kb20uZGVmYXVsdF9ybmcoYS5zZWVkKQpkZWYgdHJhaW5fbW9kZSgpOgogICAgbW9kZWwudHJhaW4oKQogICAgZm9yIG0gaW4gQk46ICAjIGJhZ3Mgb2YgNC13aW5kb3cgY2h1bmtzIGZyb20gb25lIHN0dWR5OiBrZWVwIEltYWdlTmV0IEJOIHN0YXRpc3RpY3MgZnJvemVuCiAgICAgICAgbS5ldmFsKCkKCgp0cmFpbl9tb2RlKCkKaWYgdGVhY2hlciBpcyBub3QgTm9uZToKICAgIHRlYWNoZXIuZXZhbCgpCnN0ZXAgPSBtaWNybyA9IDAKdF9wcm9iZSA9IE5vbmUKZXBvY2hzID0gYS5lbWF4CmJ1ZGdldF9zID0gYS5idWRnZXRfaCAqIDM2MDAKZm9yIGVwIGluIHJhbmdlKGEuZW1heCk6CiAgICBpZiBlcCA+PSBlcG9jaHM6CiAgICAgICAgYnJlYWsKICAgIHRvdCwgbiwgdF9lcCA9IDAuMCwgMCwgdGltZS50aW1lKCkKICAgIGZvciB0cmksIGNzLCB5IGluIGRsOgogICAgICAgIHRyaSwgeSA9IHRyaS50byhkZXYsIG5vbl9ibG9ja2luZz1UcnVlKSwgeS50byhkZXYsIG5vbl9ibG9ja2luZz1UcnVlKQogICAgICAgIGNzID0gY3MubnVtcHkoKQogICAgICAgIGF1ZyA9IChybmcucmFuZG9tKDEpIDwgMC41LCBybmcuaW50ZWdlcnMoLTEyLCAxMywgKDEsIDIpKSwgcm5nLnVuaWZvcm0oMC45LCAxLjEsIDEpLCBybmcudW5pZm9ybSgwLjksIDEuMSwgMSkpCiAgICAgICAgd2l0aCB0b3JjaC5hdXRvY2FzdChkZXYudHlwZSwgZHR5cGU9dG9yY2guZmxvYXQxNiwgZW5hYmxlZD1kZXYudHlwZSA9PSAiY3VkYSIpOgogICAgICAgICAgICBsb2dpdHMsIHhnID0gZm9yd2FyZCh0cmksIGNzLCBUcnVlLCBhdWcpCiAgICAgICAgICAgIGlmIHRlYWNoZXIgaXMgbm90IE5vbmU6CiAgICAgICAgICAgICAgICB3aXRoIHRvcmNoLm5vX2dyYWQoKToKICAgICAgICAgICAgICAgICAgICB0ID0gdG9yY2guc2lnbW9pZCh0ZWFjaGVyLmhlYWQoZW5jb2RlKHRlYWNoZXIsIHhnLmhhbGYoKSwgRmFsc2UpKS5mbG9hdCgpKQogICAgICAgICAgICAgICAgdGd0ID0gdG9yY2gud2hlcmUodG9yY2guaXNuYW4oeSksIHQsIGEuYWxwaGEgKiB5ICsgKDEgLSBhLmFscGhhKSAqIHQpCiAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICB0Z3QgPSB0b3JjaC5uYW5fdG9fbnVtKHksIG5hbj0wLjApCiAgICAgICAgdyA9ICh+dG9yY2guaXNuYW4odGd0IGlmIHRlYWNoZXIgaXMgbm90IE5vbmUgZWxzZSB5KSkuZmxvYXQoKSAgIyB1bmtub3duIGxhYmVsIGNlbGxzIGNhcnJ5IG5vIGxvc3MKICAgICAgICBsb3NzID0gKEYuYmluYXJ5X2Nyb3NzX2VudHJvcHlfd2l0aF9sb2dpdHMobG9naXRzLmZsb2F0KCksIHRndCwgcG9zX3dlaWdodD1QVywgcmVkdWN0aW9uPSJub25lIikgKiB3KS5zdW0oKSBcCiAgICAgICAgICAgIC8gdy5zdW0oKS5jbGFtcF9taW4oMSkgLyBhLmFjY3VtCiAgICAgICAgc2NhbGVyLnNjYWxlKGxvc3MpLmJhY2t3YXJkKCkKICAgICAgICBtaWNybyArPSAxCiAgICAgICAgdG90ICs9IGxvc3MuaXRlbSgpICogYS5hY2N1bTsgbiArPSAxCiAgICAgICAgaWYgbWljcm8gJSBhLmFjY3VtID09IDA6CiAgICAgICAgICAgIHNldF9scihzdGVwKQogICAgICAgICAgICBzY2FsZXIudW5zY2FsZV8ob3B0KQogICAgICAgICAgICB0b3JjaC5ubi51dGlscy5jbGlwX2dyYWRfbm9ybV8obW9kZWwucGFyYW1ldGVycygpLCAzLjApCiAgICAgICAgICAgIHNjYWxlci5zdGVwKG9wdCk7IHNjYWxlci51cGRhdGUoKTsgb3B0Lnplcm9fZ3JhZChzZXRfdG9fbm9uZT1UcnVlKQogICAgICAgICAgICBzdGVwICs9IDEKICAgICAgICBpZiBtaWNybyA9PSA0MCBhbmQgdF9wcm9iZSBpcyBOb25lOgogICAgICAgICAgICB0X3Byb2JlID0gdGltZS50aW1lKCkKICAgICAgICBpZiBtaWNybyA9PSAxMjAgYW5kIHBsYW5bInRvdGFsIl0gPT0gMTAgKiogOToKICAgICAgICAgICAgcGVyID0gKHRpbWUudGltZSgpIC0gdF9wcm9iZSkgLyA4MCAgICAgICAgICAgICAgICAgICAgICAjIHNlY29uZHMgcGVyIHN0dWR5CiAgICAgICAgICAgIGVwX3MgPSBwZXIgKiBsZW4odHJhaW4pICsgMjQwICogKDIgaWYgYS5hcm0gPT0gIkIiIGVsc2UgMSkKICAgICAgICAgICAgbGVmdCA9IGJ1ZGdldF9zIC0gKHRpbWUudGltZSgpIC0gVDApIC0gNjAwCiAgICAgICAgICAgIGVwb2NocyA9IGludChtYXgoMSwgbWluKGEuZW1heCwgKGxlZnQgKyBwZXIgKiAxMjApIC8vIGVwX3MpKSkKICAgICAgICAgICAgcGxhblsidG90YWwiXSA9IGVwb2NocyAqIChsZW4odHJhaW4pIC8vIGEuYWNjdW0pCiAgICAgICAgICAgIGxvZyhldj0icGxhbiIsIHNlY19wZXJfc3R1ZHk9cm91bmQocGVyLCAzKSwgZXBvY2hfbWluPXJvdW5kKGVwX3MgLyA2MCwgMSksIGVwb2Nocz1lcG9jaHMsCiAgICAgICAgICAgICAgICBvcHRfc3RlcHM9cGxhblsidG90YWwiXSwgbWVtX2diPXJvdW5kKHRvcmNoLmN1ZGEubWF4X21lbW9yeV9hbGxvY2F0ZWQoKSAvIDIqKjMwLCAyKSBpZiBkZXYudHlwZSA9PSAiY3VkYSIgZWxzZSAwKQogICAgICAgIGlmIG1pY3JvICUgMjAwID09IDA6CiAgICAgICAgICAgIGxvZyhldj0ic3RlcCIsIGVwPWVwLCBtaWNybz1taWNybywgc3RlcD1zdGVwLCBsb3NzPXJvdW5kKHRvdCAvIG1heChuLCAxKSwgNCksCiAgICAgICAgICAgICAgICBscj1bcm91bmQoZ1sibHIiXSwgOCkgZm9yIGcgaW4gb3B0LnBhcmFtX2dyb3Vwc10sIHNjYWxlPXNjYWxlci5nZXRfc2NhbGUoKSkKICAgICAgICBpZiB0aW1lLnRpbWUoKSAtIFQwID4gYnVkZ2V0X3M6CiAgICAgICAgICAgIGJyZWFrCiAgICBsb2coZXY9ImVwb2NoIiwgZXA9ZXAsIGxvc3M9cm91bmQodG90IC8gbWF4KG4sIDEpLCA0KSwgbWluPXJvdW5kKCh0aW1lLnRpbWUoKSAtIHRfZXApIC8gNjAsIDEpKQogICAgZyA9IGdvbGRfZXZhbChmImVwe2VwfSIpCiAgICBzYXZlKGYiZXB7ZXB9IiwgeyJlcG9jaCI6IGVwLCAiZ29sZF9tYWNybyI6IGd9KQogICAgaWYgdGltZS50aW1lKCkgLSBUMCA+IGJ1ZGdldF9zOgogICAgICAgIGxvZyhldj0iYnVkZ2V0X3N0b3AiLCBlcD1lcCkKICAgICAgICBicmVhawpzYXZlKCJsYXN0IiwgeyJlcG9jaCI6IGVwfSkKbG9nKGV2PSJkb25lIiwgZXBvY2hzX3J1bj1lcCArIDEsIHBsYW5uZWQ9ZXBvY2hzLCBob3Vycz1yb3VuZCgodGltZS50aW1lKCkgLSBUMCkgLyAzNjAwLCAyKSkK"
PARENT = "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
PARENT_SHA = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
status = {"stage": "start"}


def say(*x):
    print(f"[{time.time() - T0:7.0f}s]", *x, flush=True)


def put(**kw):
    status.update(kw, elapsed_h=round((time.time() - T0) / 3600, 3))
    json.dump(status, open(f"{W}/status.json", "w"), indent=1)


def find(name, root="/kaggle/input"):
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if "abnormality-detection" not in x and x != "competitions"]
        if name in fs:
            return os.path.join(d, name)
    return None


def comp_root():
    for b in ["/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"]:
        if os.path.exists(b + "/train.csv"):
            return b
    raise RuntimeError("competition data not attached")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------- 1. inputs
put(stage="inputs")
COMP = comp_root()
core_src = find("zoo_raptor_core.py")
parent = find(PARENT)
assert core_src and parent, (core_src, parent)
assert sha(parent) == PARENT_SHA, "parent checkpoint bytes differ"
os.makedirs(f"{W}/core", exist_ok=True)
shutil.copy(core_src, f"{W}/core/zoo_raptor_core.py")
open(f"{W}/core/train_ft.py", "wb").write(base64.b64decode(TRAIN_B64))
say("comp", COMP, "| core", core_src, "| parent ok")

# ---------------------------------------------------------------- 2. labels
tr = pd.read_csv(f"{COMP}/train.csv")
tr["StudyInstanceUID"] = tr.StudyInstanceUID.astype(str)
ids = tr.StudyInstanceUID
is_gold = tr[LAB].notna().all(axis=1).values
gold = ids[is_gold].tolist()


def table(fname, suffix=""):
    p = find(fname)
    if p is None:
        say("label source missing:", fname)
        return None
    try:
        d = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        if "StudyInstanceUID" not in d.columns:
            d = d.reset_index().rename(columns={d.index.name or "index": "StudyInstanceUID"})
        d["StudyInstanceUID"] = d.StudyInstanceUID.astype(str)
        t = d.set_index("StudyInstanceUID")[[c + suffix for c in LAB]].set_axis(LAB, axis=1).astype(float)
    except Exception as e:
        say("label source unreadable, skipped:", fname, type(e).__name__, str(e)[:120])
        return None
    t = t.where((t >= 0) & (t <= 1))
    return t[~t.index.duplicated()].reindex(ids)


src = {"steven_v4": table("llm_labels_v4_blend.csv"), "pilkwang_v2": table("report_labels_v2.csv"),
       "lixin_gpt56": table("labels_llm_gpt56sol.csv"), "qwen": table("qwen_knee_weak_labels.csv", "__label"),
       "nartaa_v1_blend": table("labels_v1_blend.parquet")}
src = {k: v for k, v in src.items() if v is not None}
stack = np.stack([v.values for v in src.values()])               # [S, N, 12]
cons = np.clip(np.nanmean(stack, 0), 0.01, 0.99)
cover = {k: round(float(v.notna().mean().mean()), 3) for k, v in src.items()}
say("label sources", cover)
assert len(src) >= 3, "need at least three label sources"
y = {u: [None if np.isnan(x) else round(float(x), 4) for x in cons[i]] for i, u in enumerate(ids) if not is_gold[i]}
gold_y = {u: tr.loc[tr.StudyInstanceUID == u, LAB].values[0].astype(float).tolist() for u in gold}
json.dump({"y": y, "gold": gold, "gold_y": gold_y, "sources": list(src)}, open(f"{W}/labels_k1.json", "w"))
put(stage="labels", label_sources=cover, consensus_mean={c: round(float(np.nanmean(cons[~is_gold, j])), 3) for j, c in enumerate(LAB)})

# ---------------------------------------------------------------- 3. parity gate
sys.path.insert(0, f"{W}/core")
import zoo_raptor_core as C  # noqa: E402

ser = pd.read_csv(f"{COMP}/train_series.csv")
ser["StudyInstanceUID"] = ser.StudyInstanceUID.astype(str); ser["SeriesInstanceUID"] = ser.SeriesInstanceUID.astype(str)
SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
tsdir = f"{COMP}/train_series" if os.path.isdir(f"{COMP}/train_series") else f"{COMP}/train_images"
reader = C._load_fastread().make_builder(C.FASTREAD_VARIANT, C.IMG, C.SLOTS, hdr_threads=C.FR_HDR, px_threads=C.FR_PX)
par = []
for s in range(3):
    idx = pd.read_csv(find(f"index_s{s}.csv"))
    vol = np.load(find(f"vol_s{s}.u8.npy"), mmap_mode="r")
    r = idx.iloc[len(idx) // 2]
    v, m = reader(str(r.StudyInstanceUID), SER, tsdir)
    lo = (C.IMG - vol.shape[-1]) // 2
    diff = int(np.abs(v[:, lo:lo + vol.shape[-1], lo:lo + vol.shape[-1]].astype(int) - vol[int(r.row)].astype(int)).max())
    par.append(diff)
say("parity max|diff| per shard sample:", par)
put(stage="parity", parity=par)
assert max(par) == 0, f"tensor/reader mismatch {par}"
del reader, C

# ---------------------------------------------------------------- 4. arms
ARMS = {"A": dict(gpu=0, extra=["--view", "zoom", "--zoom", "0.6"]), "B": dict(gpu=1, extra=["--view", "global"])}
procs = {}
for arm, cfg in ARMS.items():
    out = f"{W}/arm{arm}"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, f"{W}/core/train_ft.py", "--arm", "A", "--core", f"{W}/core", "--parent", "", "--pretrained", "--arch", "tf_efficientnetv2_s.in21k_ft_in1k", "--alpha", "1.0", "--bb_lr", "2e-4", "--head_lr", "1e-3", "--emax", "16", "--chunk", "12",
           "--labels_json", f"{W}/labels_k1.json", "--out", out, "--budget_h", str(BUDGET_H)] + cfg["extra"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu"]), PYTHONUNBUFFERED="1")
    procs[arm] = subprocess.Popen(cmd, stdout=open(f"{out}/stdout.txt", "w"), stderr=subprocess.STDOUT, env=env)
    say("launched arm", arm, "gpu", cfg["gpu"])
put(stage="training", arms=list(ARMS))


def tail(arm, n=3):
    try:
        return open(f"{W}/arm{arm}/stdout.txt").read().strip().splitlines()[-n:]
    except Exception:
        return []


while any(p.poll() is None for p in procs.values()):
    time.sleep(120)
    put(stage="training", rc={a: p.poll() for a, p in procs.items()}, tail={a: tail(a, 2) for a in procs})
    for a in procs:
        for line in tail(a, 1):
            say(a, line[:300])
    if time.time() - T0 > HARD_STOP_H * 3600:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        say("hard stop")
        break
for a, p in procs.items():
    p.wait(timeout=300)
put(stage="done", rc={a: p.returncode for a, p in procs.items()}, tail={a: tail(a, 6) for a in procs},
    files=sorted(os.path.relpath(f, W) for f in glob.glob(f"{W}/arm*/*.pt")))
say("done", status["rc"])
