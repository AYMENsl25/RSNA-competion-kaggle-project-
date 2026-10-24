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
BUDGET_H = float(os.environ.get("K1_BUDGET_H", "8.0"))
TRAIN_B64 = "IiIiSzE6IGZpbmUtdHVuZSBuYXJ0YWEncyBwdWJsaWMgUmFwdG9yLTM4NCBDb0F0TmV0IChMQiAwLjk0OSkgb24gbG9ubmllcWluJ3MgOTYtc2xpY2UgdGVuc29ycy4KClRoZSB0ZW5zb3JzIHdlcmUgYnVpbHQgYnkgdGhlIHNhbWUgZW1iZWRkZWQgRElDT00gcmVhZGVyIHRoZSAwLjk1OCBub3RlYm9vayB1c2VzIGF0IHRlc3QgdGltZSAoZGVjb2RlZApzb3VyY2Ugc2hhIDVhNTliNmY4Li4uKSwgc3RvcmVkIGFzIHRoZSBjZW50cmUgMzUyIHB4IG9mIHRoZSAzODQgcHggLyAxNDAgbW0gdm9sdW1lLCBzbyB0aGUgMzIwIHB4IGNlbnRyZQpjcm9wIGhlcmUgaXMgdGhlIGluZmVyZW5jZSBpbnB1dC4gT25lIGFybSBwZXIgR1BVOgoKICBBICBhcmNoaXRlY3R1cmUgdW5jaGFuZ2VkLiBUYXJnZXRzID0gYWxwaGEgKiBjb25zZW5zdXMgb2YgNSBwdWJsaWMgTExNIHJlcG9ydC1sYWJlbCB0YWJsZXMKICAgICArICgxIC0gYWxwaGEpICogZnJvemVuIHBhcmVudCBwcmVkaWN0aW9uIG9uIHRoZSBzYW1lIGF1Z21lbnRlZCB3aW5kb3dzIChrZWVwcyB3aGF0IHRoZSBwYXJlbnQKICAgICBsZWFybmVkIGZyb20gT0FJKS4gTWlycm9yIC8gY3JvcC1zaGlmdCAvIGdhaW4gLyBnYW1tYSBhdWdtZW50YXRpb24uCiAgQiAgc2FtZSB0YXJnZXRzLCBkdWFsIHZpZXc6IHRoZSBnbG9iYWwgMzIwIGNyb3AgcGx1cyBhIGNlbnRyZSB6b29tIChmcmFjIG9mIHRoZSAzNTIgcHgsIHJlc2l6ZWQgdG8KICAgICAzMjApIG9mIHRoZSBzYW1lIHdpbmRvd3MsIHplcm8taW5pdGlhbGlzZWQgKHZpZXcsIHNsb3QpIGVtYmVkZGluZ3MgYW5kIGEgbGVhcm5lZCBhdHRlbnRpb24gb2Zmc2V0CiAgICAgZm9yIHpvb20gd2luZG93cyAoc3RhcnRzIGF0IC0yLCBzbyB0aGUgc3RhcnQgcG9pbnQgaXMgY2xvc2UgdG8gdGhlIHBhcmVudCkuCgpHb2xkLTU4IGlzIG5ldmVyIHRyYWluZWQgb247IGl0cyBBVUMgaXMgcHJpbnRlZCBwZXIgZXBvY2ggZm9yIGluZm9ybWF0aW9uIG9ubHkgKHRoZSBwYXJlbnQncyBTV0EgZXBvY2hzCndlcmUgcGlja2VkIG9uIGdvbGQtNTgsIHNvIHRoZSBwYXJlbnQncyBvd24gZ29sZCBzY29yZSBpcyBvcHRpbWlzdGljKS4gVGhlIHByZWRlY2xhcmVkIHJlc3VsdCBpcyB0aGUKbGFzdCBlcG9jaDsgdGhlIHB1YmxpYyBMQiBkZWNpZGVzLgoiIiIKaW1wb3J0IG9zLCBzeXMsIGpzb24sIHRpbWUsIG1hdGgsIGFyZ3BhcnNlLCBjb3B5CmltcG9ydCBudW1weSBhcyBucCwgcGFuZGFzIGFzIHBkCmltcG9ydCB0b3JjaCwgdG9yY2gubm4gYXMgbm4sIHRvcmNoLm5uLmZ1bmN0aW9uYWwgYXMgRgpmcm9tIHRvcmNoLnV0aWxzLmNoZWNrcG9pbnQgaW1wb3J0IGNoZWNrcG9pbnQKZnJvbSBza2xlYXJuLm1ldHJpY3MgaW1wb3J0IHJvY19hdWNfc2NvcmUKCmFwID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQphcC5hZGRfYXJndW1lbnQoIi0tYXJtIiwgY2hvaWNlcz1bIkEiLCAiQiJdLCByZXF1aXJlZD1UcnVlKQphcC5hZGRfYXJndW1lbnQoIi0tY29yZSIsIHJlcXVpcmVkPVRydWUsIGhlbHA9ImRpcmVjdG9yeSBob2xkaW5nIHpvb19yYXB0b3JfY29yZS5weSIpCmFwLmFkZF9hcmd1bWVudCgiLS1wYXJlbnQiLCBkZWZhdWx0PSIiLCBoZWxwPSJwYXJlbnQgY2hlY2twb2ludDsgZW1wdHkgPSByYW5kb20gaW5pdCAoc21va2Ugb25seSkiKQphcC5hZGRfYXJndW1lbnQoIi0tYXJjaCIsIGRlZmF1bHQ9ImNvYXRuZXRfcm1scF8yX3J3XzM4NC5zd19pbjEya19mdF9pbjFrIikKYXAuYWRkX2FyZ3VtZW50KCItLWRhdGFfcm9vdCIsIGRlZmF1bHQ9Ii9rYWdnbGUvaW5wdXQiKQphcC5hZGRfYXJndW1lbnQoIi0tbGFiZWxzX2pzb24iLCByZXF1aXJlZD1UcnVlLCBoZWxwPSJqc29uIHdyaXR0ZW4gYnkgcnVuLnB5OiB7dWlkOiBbMTIgZmxvYXRzIG9yIG51bGxdfSwgZ29sZCBsaXN0IikKYXAuYWRkX2FyZ3VtZW50KCItLW91dCIsIHJlcXVpcmVkPVRydWUpCmFwLmFkZF9hcmd1bWVudCgiLS1idWRnZXRfaCIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9OC4wKQphcC5hZGRfYXJndW1lbnQoIi0tZW1heCIsIHR5cGU9aW50LCBkZWZhdWx0PTYpCmFwLmFkZF9hcmd1bWVudCgiLS1UIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MTIpCmFwLmFkZF9hcmd1bWVudCgiLS1hY2N1bSIsIHR5cGU9aW50LCBkZWZhdWx0PTE2KQphcC5hZGRfYXJndW1lbnQoIi0tY2h1bmsiLCB0eXBlPWludCwgZGVmYXVsdD00KQphcC5hZGRfYXJndW1lbnQoIi0tYmJfbHIiLCB0eXBlPWZsb2F0LCBkZWZhdWx0PTEuNWUtNSkKYXAuYWRkX2FyZ3VtZW50KCItLWhlYWRfbHIiLCB0eXBlPWZsb2F0LCBkZWZhdWx0PTNlLTQpCmFwLmFkZF9hcmd1bWVudCgiLS1uZXdfbHIiLCB0eXBlPWZsb2F0LCBkZWZhdWx0PTFlLTMpCmFwLmFkZF9hcmd1bWVudCgiLS13ZCIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MC4wNSkKYXAuYWRkX2FyZ3VtZW50KCItLWFscGhhIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0wLjUpCmFwLmFkZF9hcmd1bWVudCgiLS16b29tIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0wLjYpCmFwLmFkZF9hcmd1bWVudCgiLS1rX2V2YWwiLCB0eXBlPWludCwgZGVmYXVsdD05NCkKYXAuYWRkX2FyZ3VtZW50KCItLXdvcmtlcnMiLCB0eXBlPWludCwgZGVmYXVsdD0yKQphcC5hZGRfYXJndW1lbnQoIi0tbGltaXQiLCB0eXBlPWludCwgZGVmYXVsdD0wLCBoZWxwPSJzbW9rZTogdHJhaW4gb24gTiBzdHVkaWVzIikKYXAuYWRkX2FyZ3VtZW50KCItLWdvbGRfbGltaXQiLCB0eXBlPWludCwgZGVmYXVsdD0wKQphcC5hZGRfYXJndW1lbnQoIi0tc2VlZCIsIHR5cGU9aW50LCBkZWZhdWx0PTcpCmEgPSBhcC5wYXJzZV9hcmdzKCkKCnN5cy5wYXRoLmluc2VydCgwLCBhLmNvcmUpCmltcG9ydCB6b29fcmFwdG9yX2NvcmUgYXMgQyAgIyBub3FhOiBFNDAyCgp0b3JjaC5tYW51YWxfc2VlZChhLnNlZWQpOyBucC5yYW5kb20uc2VlZChhLnNlZWQpCmRldiA9IHRvcmNoLmRldmljZSgiY3VkYSIgaWYgdG9yY2guY3VkYS5pc19hdmFpbGFibGUoKSBlbHNlICJjcHUiKQpvcy5tYWtlZGlycyhhLm91dCwgZXhpc3Rfb2s9VHJ1ZSkKVDAgPSB0aW1lLnRpbWUoKQpMQUIgPSBDLkxBQgpEID0gQy5NQVhTCgoKZGVmIGxvZygqKmt3KToKICAgIGt3WyJ0Il0gPSByb3VuZCh0aW1lLnRpbWUoKSAtIFQwLCAxKQogICAgbGluZSA9IGpzb24uZHVtcHMoa3cpCiAgICBwcmludChsaW5lLCBmbHVzaD1UcnVlKQogICAgd2l0aCBvcGVuKG9zLnBhdGguam9pbihhLm91dCwgImxvZy5qc29ubCIpLCAiYSIpIGFzIGY6CiAgICAgICAgZi53cml0ZShsaW5lICsgIlxuIikKCgojIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0gZGF0YQpkZWYgZmluZChuYW1lKToKICAgIGZvciBkLCBkaXJzLCBmcyBpbiBvcy53YWxrKGEuZGF0YV9yb290KToKICAgICAgICBkaXJzWzpdID0gW3ggZm9yIHggaW4gZGlycyBpZiAiYWJub3JtYWxpdHktZGV0ZWN0aW9uIiBub3QgaW4geCBhbmQgeCAhPSAiY29tcGV0aXRpb25zIl0KICAgICAgICBpZiBuYW1lIGluIGZzOgogICAgICAgICAgICByZXR1cm4gb3MucGF0aC5qb2luKGQsIG5hbWUpCiAgICByYWlzZSBGaWxlTm90Rm91bmRFcnJvcihuYW1lKQoKClZPTFMsIE1BU0tTLCBMT0MgPSBbXSwgW10sIHt9CmZvciBzIGluIHJhbmdlKDQpOgogICAgVk9MUy5hcHBlbmQobnAubG9hZChmaW5kKGYidm9sX3N7c30udTgubnB5IiksIG1tYXBfbW9kZT0iciIpKQogICAgTUFTS1MuYXBwZW5kKG5wLmxvYWQoZmluZChmIm1hc2tfc3tzfS51OC5ucHkiKSkpCiAgICBpZHggPSBwZC5yZWFkX2NzdihmaW5kKGYiaW5kZXhfc3tzfS5jc3YiKSkKICAgIGZvciB1aWQsIHJvdywgZG9uZSBpbiB6aXAoaWR4LlN0dWR5SW5zdGFuY2VVSUQuYXN0eXBlKHN0ciksIGlkeC5yb3csIGlkeC5kb25lKToKICAgICAgICBpZiBkb25lOgogICAgICAgICAgICBMT0NbdWlkXSA9IChzLCBpbnQocm93KSkKUyA9IFZPTFNbMF0uc2hhcGVbLTFdCmFzc2VydCBWT0xTWzBdLnNoYXBlWzFdID09IEQgYW5kIFMgPT0gMzUyLCBWT0xTWzBdLnNoYXBlCgpsYWIgPSBqc29uLmxvYWQob3BlbihhLmxhYmVsc19qc29uKSkKZ29sZCA9IFt1IGZvciB1IGluIGxhYlsiZ29sZCJdIGlmIHUgaW4gTE9DXQpnb2xkX3kgPSB7dTogbnAuYXJyYXkobGFiWyJnb2xkX3kiXVt1XSwgbnAuZmxvYXQzMikgZm9yIHUgaW4gZ29sZH0KdHJhaW4gPSBbdSBmb3IgdSwgdiBpbiBsYWJbInkiXS5pdGVtcygpIGlmIHUgaW4gTE9DIGFuZCB1IG5vdCBpbiBzZXQobGFiWyJnb2xkIl0pXQpZID0ge3U6IG5wLmFycmF5KFtucC5uYW4gaWYgeCBpcyBOb25lIGVsc2UgeCBmb3IgeCBpbiBsYWJbInkiXVt1XV0sIG5wLmZsb2F0MzIpIGZvciB1IGluIHRyYWlufQppZiBhLmxpbWl0OgogICAgdHJhaW4gPSB0cmFpbls6YS5saW1pdF0KaWYgYS5nb2xkX2xpbWl0OgogICAgZ29sZCA9IGdvbGRbOmEuZ29sZF9saW1pdF0KcHJldiA9IG5wLmNsaXAobnAubmFubWVhbihucC5zdGFjayhbWVt1XSBmb3IgdSBpbiB0cmFpbl0pLCAwKSwgMC4wMywgMC43KQpQVyA9IHRvcmNoLnRlbnNvcihucC5jbGlwKCgxIC0gcHJldikgLyBwcmV2LCAxLCAxMCksIGR0eXBlPXRvcmNoLmZsb2F0MzIsIGRldmljZT1kZXYpCmxvZyhldj0iZGF0YSIsIGFybT1hLmFybSwgdHJhaW49bGVuKHRyYWluKSwgZ29sZD1sZW4oZ29sZCksIHNoYXJkcz1bdi5zaGFwZVswXSBmb3IgdiBpbiBWT0xTXSwgcHc9UFcudG9saXN0KCkpCgoKY2xhc3MgQmFncyh0b3JjaC51dGlscy5kYXRhLkRhdGFzZXQpOgogICAgZGVmIF9faW5pdF9fKHNlbGYsIHVpZHMpOgogICAgICAgIHNlbGYudWlkcyA9IHVpZHMKCiAgICBkZWYgX19sZW5fXyhzZWxmKToKICAgICAgICByZXR1cm4gbGVuKHNlbGYudWlkcykKCiAgICBkZWYgX19nZXRpdGVtX18oc2VsZiwgaSk6CiAgICAgICAgdSA9IHNlbGYudWlkc1tpXQogICAgICAgIHMsIHIgPSBMT0NbdV0KICAgICAgICBybmcgPSBucC5yYW5kb20uZGVmYXVsdF9ybmcoKHRvcmNoLmluaXRpYWxfc2VlZCgpICsgaSkgJSAyKiozMikKICAgICAgICBjcyA9IEMudHJhaW5fY2VudGVycyhNQVNLU1tzXVtyXSwgRCwgYS5ULCBybmcpCiAgICAgICAgdHJpID0gbnAuYXNjb250aWd1b3VzYXJyYXkoVk9MU1tzXVtyXVtucC5zdGFjayhbY3MgLSAxLCBjcywgY3MgKyAxXSwgMSldKSAgIyBbVCwzLFMsU10gdWludDgKICAgICAgICByZXR1cm4gdG9yY2guZnJvbV9udW1weSh0cmkpLCB0b3JjaC5mcm9tX251bXB5KGNzKSwgdG9yY2guZnJvbV9udW1weShZW3VdKQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBtb2RlbApkZWYgbG9hZF9zdGF0ZShwYXRoKToKICAgIGNrID0gdG9yY2gubG9hZChwYXRoLCBtYXBfbG9jYXRpb249ImNwdSIsIHdlaWdodHNfb25seT1GYWxzZSkKICAgIHJldHVybiBja1sibW9kZWwiXQoKCnBhcmVudF9zZCA9IGxvYWRfc3RhdGUoYS5wYXJlbnQpIGlmIGEucGFyZW50IGVsc2UgTm9uZQpzdHVkZW50ID0gQy5idWlsZF9tb2RlbChhcmNoPWEuYXJjaCwgc3RhdGU9cGFyZW50X3NkKQp0ZWFjaGVyID0gY29weS5kZWVwY29weShzdHVkZW50KS5oYWxmKCkuZXZhbCgpLnRvKGRldikgaWYgcGFyZW50X3NkIGlzIG5vdCBOb25lIGVsc2UgTm9uZQppZiB0ZWFjaGVyIGlzIG5vdCBOb25lOgogICAgZm9yIHAgaW4gdGVhY2hlci5wYXJhbWV0ZXJzKCk6CiAgICAgICAgcC5yZXF1aXJlc19ncmFkXyhGYWxzZSkKU0xPVCA9IHRvcmNoLmZyb21fbnVtcHkoQy5fc2xvdF9pbmRleChEKVswXSkubG9uZygpLnRvKGRldikgICMgc2xpY2UgLT4gc2xvdCAwLi40CgoKY2xhc3MgRHVhbFZpZXcobm4uTW9kdWxlKToKICAgICIiIlJhcHRvciBoZWFkIG92ZXIgZ2xvYmFsICsgem9vbSB3aW5kb3dzOyB6ZXJvICh2aWV3LCBzbG90KSBlbWJlZGRpbmdzIGtlZXAgdGhlIHBhcmVudCBmdW5jdGlvbiBvbiB0aGUKICAgIGdsb2JhbCB2aWV3IGF0IHN0ZXAgMCwgem9vbSB3aW5kb3dzIHN0YXJ0IHdpdGggYW4gYXR0ZW50aW9uIG9mZnNldCBvZiAtMi4iIiIKCiAgICBkZWYgX19pbml0X18oc2VsZiwgYmFzZSk6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgc2VsZi5iYXNlID0gYmFzZQogICAgICAgIGZkID0gYmFzZS5jbHNXLnNoYXBlWzFdCiAgICAgICAgc2VsZi5lbWIgPSBubi5QYXJhbWV0ZXIodG9yY2guemVyb3MoMTAsIGZkKSkKICAgICAgICBzZWxmLnpiaWFzID0gbm4uUGFyYW1ldGVyKHRvcmNoLmZ1bGwoKGJhc2UubiwpLCAtMi4wKSkKCiAgICBkZWYgcG9vbChzZWxmLCBmZywgZnosIHNsb3QpOgogICAgICAgIGYgPSB0b3JjaC5jYXQoW2ZnICsgc2VsZi5lbWJbc2xvdF0sIGZ6ICsgc2VsZi5lbWJbc2xvdCArIDVdXSwgMSkKICAgICAgICBoID0gc2VsZi5iYXNlLm5vcm0oZikKICAgICAgICBhdHQgPSBzZWxmLmJhc2UuYXR0KGgpCiAgICAgICAgayA9IGZnLnNoYXBlWzFdCiAgICAgICAgYXR0ID0gdG9yY2guY2F0KFthdHRbOiwgOmtdLCBhdHRbOiwgazpdICsgc2VsZi56Ymlhc10sIDEpLmZsb2F0KCkuc29mdG1heCgxKS50byhoLmR0eXBlKQogICAgICAgIHBvb2xlZCA9IHRvcmNoLmVpbnN1bSgiYmtuLGJrZi0+Ym5mIiwgYXR0LCBoKQogICAgICAgIHJldHVybiAocG9vbGVkICogc2VsZi5iYXNlLmNsc1cpLnN1bSgtMSkgKyBzZWxmLmJhc2UuY2xzYgoKCm1vZGVsID0gKER1YWxWaWV3KHN0dWRlbnQpIGlmIGEuYXJtID09ICJCIiBlbHNlIHN0dWRlbnQpLnRvKGRldikKCgpkZWYgZW5jb2RlKG0sIHgsIHRyYWluKToKICAgIGIsIGsgPSB4LnNoYXBlWzoyXQogICAgZmxhdCA9IHguZmxhdHRlbigwLCAxKQogICAgaWYgdHJhaW46CiAgICAgICAgZiA9IHRvcmNoLmNhdChbY2hlY2twb2ludChtLmJhY2tib25lLCBjLCB1c2VfcmVlbnRyYW50PUZhbHNlKSBmb3IgYyBpbiBmbGF0LnNwbGl0KGEuY2h1bmspXSkKICAgIGVsc2U6CiAgICAgICAgZiA9IHRvcmNoLmNhdChbbS5iYWNrYm9uZShjKSBmb3IgYyBpbiBmbGF0LnNwbGl0KDE2KV0pCiAgICByZXR1cm4gZi52aWV3KGIsIGssIC0xKQoKCmRlZiB6b29tX3ZpZXdzKHRyaSwgY3MsIG1pcnJvciwgb2ZmcywgZ2FpbiwgZ2FtbWEpOgogICAgIiIiQ2VudHJlIHpvb20gb2YgdGhlIHNhbWUgdWludDggd2luZG93czogY3JvcCBmcmFjKlMgKHNoaWZ0ZWQgbGlrZSB0aGUgZ2xvYmFsIGNyb3ApLCByZXNpemUgdG8gMzIwLiIiIgogICAgYiA9IHRyaS5zaGFwZVswXQogICAgeiA9IGludChyb3VuZChhLnpvb20gKiBTKSkKICAgIHJvd3MgPSBbXQogICAgZm9yIGkgaW4gcmFuZ2UoYik6CiAgICAgICAgeCA9IHRyaVtpXQogICAgICAgIGlmIG1pcnJvciBpcyBub3QgTm9uZSBhbmQgYm9vbChtaXJyb3JbaV0pOgogICAgICAgICAgICB4ID0gQy5fbWlycm9yX3RyaSh4LCBjc1tpXSwgRCkKICAgICAgICBveSwgb3ggPSAoaW50KG9mZnNbaV1bMF0pLCBpbnQob2Zmc1tpXVsxXSkpIGlmIG9mZnMgaXMgbm90IE5vbmUgZWxzZSAoMCwgMCkKICAgICAgICB5MCA9IG1pbihtYXgoKFMgLSB6KSAvLyAyICsgb3ksIDApLCBTIC0geikKICAgICAgICB4MCA9IG1pbihtYXgoKFMgLSB6KSAvLyAyICsgb3gsIDApLCBTIC0geikKICAgICAgICByb3dzLmFwcGVuZCh4WzosIDosIHkwOnkwICsgeiwgeDA6eDAgKyB6XSkKICAgIHggPSB0b3JjaC5zdGFjayhyb3dzKS5mbG9hdCgpLmRpdl8oMjU1LjApCiAgICBpZiBnYW1tYSBpcyBub3QgTm9uZToKICAgICAgICB4ID0geC5jbGFtcF9taW4oMWUtNCkucG93KHRvcmNoLmFzX3RlbnNvcihnYW1tYSwgZHR5cGU9dG9yY2guZmxvYXQzMiwgZGV2aWNlPXguZGV2aWNlKS52aWV3KGIsIDEsIDEsIDEsIDEpKQogICAgaWYgZ2FpbiBpcyBub3QgTm9uZToKICAgICAgICB4ID0gKHggKiB0b3JjaC5hc190ZW5zb3IoZ2FpbiwgZHR5cGU9dG9yY2guZmxvYXQzMiwgZGV2aWNlPXguZGV2aWNlKS52aWV3KGIsIDEsIDEsIDEsIDEpKS5jbGFtcF8oMCwgMSkKICAgIGsgPSB4LnNoYXBlWzFdCiAgICB4ID0gRi5pbnRlcnBvbGF0ZSh4LmZsYXR0ZW4oMCwgMSksIHNpemU9KEMuSU5QVVRfUkVTLCBDLklOUFVUX1JFUyksIG1vZGU9ImJpbGluZWFyIiwgYWxpZ25fY29ybmVycz1GYWxzZSkKICAgIHggPSB4LnZpZXcoYiwgaywgMywgQy5JTlBVVF9SRVMsIEMuSU5QVVRfUkVTKQogICAgcmV0dXJuICh4IC0gQy5fTUVBTi50byh4LmRldmljZSkudmlldygxLCAxLCAzLCAxLCAxKSkgLyBDLl9TVEQudG8oeC5kZXZpY2UpLnZpZXcoMSwgMSwgMywgMSwgMSkKCgpkZWYgZm9yd2FyZCh0cmksIGNzLCB0cmFpbiwgYXVnPU5vbmUpOgogICAgbWlycm9yLCBvZmZzLCBnYWluLCBnYW1tYSA9IGF1ZyBpZiBhdWcgaXMgbm90IE5vbmUgZWxzZSAoTm9uZSwgTm9uZSwgTm9uZSwgTm9uZSkKICAgIHhnID0gQy5ncHVfdmlld3ModHJpLCBjcywgRCwgQy5JTlBVVF9SRVMsIG1pcnJvcj1taXJyb3IsIG9mZnM9b2ZmcywgZ2Fpbj1nYWluLCBnYW1tYT1nYW1tYSkKICAgIGlmIGEuYXJtID09ICJBIjoKICAgICAgICByZXR1cm4gc3R1ZGVudC5oZWFkKGVuY29kZShzdHVkZW50LCB4ZywgdHJhaW4pKSwgeGcKICAgIHh6ID0gem9vbV92aWV3cyh0cmksIGNzLCBtaXJyb3IsIG9mZnMsIGdhaW4sIGdhbW1hKQogICAgc2xvdCA9IFNMT1RbdG9yY2guYXNfdGVuc29yKGNzLCBkZXZpY2U9ZGV2KS5sb25nKCldCiAgICByZXR1cm4gbW9kZWwucG9vbChlbmNvZGUoc3R1ZGVudCwgeGcsIHRyYWluKSwgZW5jb2RlKHN0dWRlbnQsIHh6LCB0cmFpbiksIHNsb3QpLCB4ZwoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBnb2xkIG1vbml0b3IgKGluZmVyZW5jZSByZWNpcGU6IGtfZXZhbCB3aW5kb3dzLCBwbGFpbiArIG1pcnJvcikKQHRvcmNoLm5vX2dyYWQoKQpkZWYgZ29sZF9ldmFsKHRhZyk6CiAgICBpZiBub3QgZ29sZDoKICAgICAgICByZXR1cm4gTm9uZQogICAgbW9kZWwuZXZhbCgpCiAgICBQLCBZZyA9IFtdLCBbXQogICAgZm9yIHUgaW4gZ29sZDoKICAgICAgICBzLCByID0gTE9DW3VdCiAgICAgICAgY3MgPSBDLmV2YWxfY2VudGVycyhNQVNLU1tzXVtyXSwgRCwgYS5rX2V2YWwpCiAgICAgICAgdHJpID0gdG9yY2guZnJvbV9udW1weShucC5hc2NvbnRpZ3VvdXNhcnJheShWT0xTW3NdW3JdW25wLnN0YWNrKFtjcyAtIDEsIGNzLCBjcyArIDFdLCAxKV0pKS50byhkZXYpW05vbmVdCiAgICAgICAgcHMgPSBbXQogICAgICAgIGZvciBtaXIgaW4gKEZhbHNlLCBUcnVlKToKICAgICAgICAgICAgd2l0aCB0b3JjaC5hdXRvY2FzdChkZXYudHlwZSwgZHR5cGU9dG9yY2guZmxvYXQxNiwgZW5hYmxlZD1kZXYudHlwZSA9PSAiY3VkYSIpOgogICAgICAgICAgICAgICAgbGcsIF8gPSBmb3J3YXJkKHRyaSwgY3NbTm9uZV0sIEZhbHNlLCAoW21pcl0sIE5vbmUsIE5vbmUsIE5vbmUpKQogICAgICAgICAgICBwcy5hcHBlbmQodG9yY2guc2lnbW9pZChsZy5mbG9hdCgpKVswXS5jcHUoKS5udW1weSgpKQogICAgICAgIFAuYXBwZW5kKG5wLm1lYW4ocHMsIDApKTsgWWcuYXBwZW5kKGdvbGRfeVt1XSkKICAgIFAsIFlnID0gbnAuc3RhY2soUCksIG5wLnN0YWNrKFlnKQogICAgYXVjcyA9IHtuOiBmbG9hdChyb2NfYXVjX3Njb3JlKFlnWzosIGpdLCBQWzosIGpdKSkgZm9yIGosIG4gaW4gZW51bWVyYXRlKExBQikgaWYgbGVuKHNldChZZ1s6LCBqXSkpID4gMX0KICAgIG1hY3JvID0gZmxvYXQobnAubWVhbihsaXN0KGF1Y3MudmFsdWVzKCkpKSkgaWYgYXVjcyBlbHNlIGZsb2F0KCJuYW4iKQogICAgbnAuc2F2ZShvcy5wYXRoLmpvaW4oYS5vdXQsIGYiZ29sZF97dGFnfS5ucHkiKSwgUCkKICAgIGxvZyhldj0iZ29sZCIsIHRhZz10YWcsIG1hY3JvPXJvdW5kKG1hY3JvLCA0KSwgcGVyPXtrOiByb3VuZCh2LCAzKSBmb3IgaywgdiBpbiBhdWNzLml0ZW1zKCl9KQogICAgbW9kZWwudHJhaW4oKQogICAgcmV0dXJuIG1hY3JvCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIG9wdGltaXNlcgpiYiA9IFtwIGZvciBuLCBwIGluIHN0dWRlbnQubmFtZWRfcGFyYW1ldGVycygpIGlmIG4uc3RhcnRzd2l0aCgiYmFja2JvbmUuIildCmhkID0gW3AgZm9yIG4sIHAgaW4gc3R1ZGVudC5uYW1lZF9wYXJhbWV0ZXJzKCkgaWYgbm90IG4uc3RhcnRzd2l0aCgiYmFja2JvbmUuIildCmdyb3VwcyA9IFt7InBhcmFtcyI6IGJiLCAibHIiOiBhLmJiX2xyfSwgeyJwYXJhbXMiOiBoZCwgImxyIjogYS5oZWFkX2xyfV0KaWYgYS5hcm0gPT0gIkIiOgogICAgZ3JvdXBzLmFwcGVuZCh7InBhcmFtcyI6IFttb2RlbC5lbWIsIG1vZGVsLnpiaWFzXSwgImxyIjogYS5uZXdfbHIsICJ3ZWlnaHRfZGVjYXkiOiAwLjB9KQpvcHQgPSB0b3JjaC5vcHRpbS5BZGFtVyhncm91cHMsIHdlaWdodF9kZWNheT1hLndkKQpiYXNlX2xyID0gW2dbImxyIl0gZm9yIGcgaW4gb3B0LnBhcmFtX2dyb3Vwc10Kc2NhbGVyID0gdG9yY2guYW1wLkdyYWRTY2FsZXIoImN1ZGEiLCBlbmFibGVkPWRldi50eXBlID09ICJjdWRhIikKV0FSTSA9IDMwCnBsYW4gPSB7InRvdGFsIjogMTAgKiogOX0KCgpkZWYgc2V0X2xyKHN0ZXApOgogICAgaWYgc3RlcCA8IFdBUk06CiAgICAgICAgbSA9IChzdGVwICsgMSkgLyBXQVJNCiAgICBlbHNlOgogICAgICAgIG0gPSAwLjUgKiAoMSArIG1hdGguY29zKG1hdGgucGkgKiBtaW4oMS4wLCAoc3RlcCAtIFdBUk0pIC8gbWF4KDEsIHBsYW5bInRvdGFsIl0gLSBXQVJNKSkpKQogICAgZm9yIGcsIGIwIGluIHppcChvcHQucGFyYW1fZ3JvdXBzLCBiYXNlX2xyKToKICAgICAgICBnWyJsciJdID0gYjAgKiBtCgoKZGVmIHNhdmUodGFnLCBleHRyYT1Ob25lKToKICAgIHNkID0ge2s6ICh2LmhhbGYoKSBpZiB2LmlzX2Zsb2F0aW5nX3BvaW50KCkgZWxzZSB2KSBmb3IgaywgdiBpbiBtb2RlbC5zdGF0ZV9kaWN0KCkuaXRlbXMoKX0KICAgIGNrID0geyJtb2RlbCI6IHNkLCAiYXJjaCI6IGEuYXJjaCwgInJlcyI6IDM4NCwgImxhYiI6IExBQiwgImFybSI6IGEuYXJtLCAiem9vbSI6IGEuem9vbSwgInRhZyI6IHRhZywKICAgICAgICAgICJwYXJlbnQiOiBvcy5wYXRoLmJhc2VuYW1lKGEucGFyZW50KSwgImFyZ3MiOiB2YXJzKGEpfQogICAgY2sudXBkYXRlKGV4dHJhIG9yIHt9KQogICAgdG9yY2guc2F2ZShjaywgb3MucGF0aC5qb2luKGEub3V0LCBmImsxe2EuYXJtfV97dGFnfS5wdC50bXAiKSkKICAgIG9zLnJlcGxhY2Uob3MucGF0aC5qb2luKGEub3V0LCBmImsxe2EuYXJtfV97dGFnfS5wdC50bXAiKSwgb3MucGF0aC5qb2luKGEub3V0LCBmImsxe2EuYXJtfV97dGFnfS5wdCIpKQoKCiMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSB0cmFpbgpnb2xkX2V2YWwoInBhcmVudCIpCmRsID0gdG9yY2gudXRpbHMuZGF0YS5EYXRhTG9hZGVyKEJhZ3ModHJhaW4pLCBiYXRjaF9zaXplPTEsIHNodWZmbGU9VHJ1ZSwgbnVtX3dvcmtlcnM9YS53b3JrZXJzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwaW5fbWVtb3J5PWRldi50eXBlID09ICJjdWRhIiwgcGVyc2lzdGVudF93b3JrZXJzPWEud29ya2VycyA+IDAsIGRyb3BfbGFzdD1UcnVlKQpybmcgPSBucC5yYW5kb20uZGVmYXVsdF9ybmcoYS5zZWVkKQptb2RlbC50cmFpbigpCmlmIHRlYWNoZXIgaXMgbm90IE5vbmU6CiAgICB0ZWFjaGVyLmV2YWwoKQpzdGVwID0gbWljcm8gPSAwCnRfcHJvYmUgPSBOb25lCmVwb2NocyA9IGEuZW1heApidWRnZXRfcyA9IGEuYnVkZ2V0X2ggKiAzNjAwCmZvciBlcCBpbiByYW5nZShhLmVtYXgpOgogICAgaWYgZXAgPj0gZXBvY2hzOgogICAgICAgIGJyZWFrCiAgICB0b3QsIG4sIHRfZXAgPSAwLjAsIDAsIHRpbWUudGltZSgpCiAgICBmb3IgdHJpLCBjcywgeSBpbiBkbDoKICAgICAgICB0cmksIHkgPSB0cmkudG8oZGV2LCBub25fYmxvY2tpbmc9VHJ1ZSksIHkudG8oZGV2LCBub25fYmxvY2tpbmc9VHJ1ZSkKICAgICAgICBjcyA9IGNzLm51bXB5KCkKICAgICAgICBhdWcgPSAocm5nLnJhbmRvbSgxKSA8IDAuNSwgcm5nLmludGVnZXJzKC0xMiwgMTMsICgxLCAyKSksIHJuZy51bmlmb3JtKDAuOSwgMS4xLCAxKSwgcm5nLnVuaWZvcm0oMC45LCAxLjEsIDEpKQogICAgICAgIHdpdGggdG9yY2guYXV0b2Nhc3QoZGV2LnR5cGUsIGR0eXBlPXRvcmNoLmZsb2F0MTYsIGVuYWJsZWQ9ZGV2LnR5cGUgPT0gImN1ZGEiKToKICAgICAgICAgICAgbG9naXRzLCB4ZyA9IGZvcndhcmQodHJpLCBjcywgVHJ1ZSwgYXVnKQogICAgICAgICAgICBpZiB0ZWFjaGVyIGlzIG5vdCBOb25lOgogICAgICAgICAgICAgICAgd2l0aCB0b3JjaC5ub19ncmFkKCk6CiAgICAgICAgICAgICAgICAgICAgdCA9IHRvcmNoLnNpZ21vaWQodGVhY2hlci5oZWFkKGVuY29kZSh0ZWFjaGVyLCB4Zy5oYWxmKCksIEZhbHNlKSkuZmxvYXQoKSkKICAgICAgICAgICAgICAgIHRndCA9IHRvcmNoLndoZXJlKHRvcmNoLmlzbmFuKHkpLCB0LCBhLmFscGhhICogeSArICgxIC0gYS5hbHBoYSkgKiB0KQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgdGd0ID0gdG9yY2gubmFuX3RvX251bSh5LCBuYW49MC4wKQogICAgICAgIGxvc3MgPSBGLmJpbmFyeV9jcm9zc19lbnRyb3B5X3dpdGhfbG9naXRzKGxvZ2l0cy5mbG9hdCgpLCB0Z3QsIHBvc193ZWlnaHQ9UFcpIC8gYS5hY2N1bQogICAgICAgIHNjYWxlci5zY2FsZShsb3NzKS5iYWNrd2FyZCgpCiAgICAgICAgbWljcm8gKz0gMQogICAgICAgIHRvdCArPSBsb3NzLml0ZW0oKSAqIGEuYWNjdW07IG4gKz0gMQogICAgICAgIGlmIG1pY3JvICUgYS5hY2N1bSA9PSAwOgogICAgICAgICAgICBzZXRfbHIoc3RlcCkKICAgICAgICAgICAgc2NhbGVyLnVuc2NhbGVfKG9wdCkKICAgICAgICAgICAgdG9yY2gubm4udXRpbHMuY2xpcF9ncmFkX25vcm1fKG1vZGVsLnBhcmFtZXRlcnMoKSwgMy4wKQogICAgICAgICAgICBzY2FsZXIuc3RlcChvcHQpOyBzY2FsZXIudXBkYXRlKCk7IG9wdC56ZXJvX2dyYWQoc2V0X3RvX25vbmU9VHJ1ZSkKICAgICAgICAgICAgc3RlcCArPSAxCiAgICAgICAgaWYgbWljcm8gPT0gNDAgYW5kIHRfcHJvYmUgaXMgTm9uZToKICAgICAgICAgICAgdF9wcm9iZSA9IHRpbWUudGltZSgpCiAgICAgICAgaWYgbWljcm8gPT0gMTIwIGFuZCBwbGFuWyJ0b3RhbCJdID09IDEwICoqIDk6CiAgICAgICAgICAgIHBlciA9ICh0aW1lLnRpbWUoKSAtIHRfcHJvYmUpIC8gODAgICAgICAgICAgICAgICAgICAgICAgIyBzZWNvbmRzIHBlciBzdHVkeQogICAgICAgICAgICBlcF9zID0gcGVyICogbGVuKHRyYWluKSArIDI0MCAqICgyIGlmIGEuYXJtID09ICJCIiBlbHNlIDEpCiAgICAgICAgICAgIGxlZnQgPSBidWRnZXRfcyAtICh0aW1lLnRpbWUoKSAtIFQwKSAtIDYwMAogICAgICAgICAgICBlcG9jaHMgPSBpbnQobWF4KDEsIG1pbihhLmVtYXgsIChsZWZ0ICsgcGVyICogMTIwKSAvLyBlcF9zKSkpCiAgICAgICAgICAgIHBsYW5bInRvdGFsIl0gPSBlcG9jaHMgKiAobGVuKHRyYWluKSAvLyBhLmFjY3VtKQogICAgICAgICAgICBsb2coZXY9InBsYW4iLCBzZWNfcGVyX3N0dWR5PXJvdW5kKHBlciwgMyksIGVwb2NoX21pbj1yb3VuZChlcF9zIC8gNjAsIDEpLCBlcG9jaHM9ZXBvY2hzLAogICAgICAgICAgICAgICAgb3B0X3N0ZXBzPXBsYW5bInRvdGFsIl0sIG1lbV9nYj1yb3VuZCh0b3JjaC5jdWRhLm1heF9tZW1vcnlfYWxsb2NhdGVkKCkgLyAyKiozMCwgMikgaWYgZGV2LnR5cGUgPT0gImN1ZGEiIGVsc2UgMCkKICAgICAgICBpZiBtaWNybyAlIDIwMCA9PSAwOgogICAgICAgICAgICBsb2coZXY9InN0ZXAiLCBlcD1lcCwgbWljcm89bWljcm8sIHN0ZXA9c3RlcCwgbG9zcz1yb3VuZCh0b3QgLyBtYXgobiwgMSksIDQpLAogICAgICAgICAgICAgICAgbHI9W3JvdW5kKGdbImxyIl0sIDgpIGZvciBnIGluIG9wdC5wYXJhbV9ncm91cHNdLCBzY2FsZT1zY2FsZXIuZ2V0X3NjYWxlKCkpCiAgICAgICAgaWYgdGltZS50aW1lKCkgLSBUMCA+IGJ1ZGdldF9zOgogICAgICAgICAgICBicmVhawogICAgbG9nKGV2PSJlcG9jaCIsIGVwPWVwLCBsb3NzPXJvdW5kKHRvdCAvIG1heChuLCAxKSwgNCksIG1pbj1yb3VuZCgodGltZS50aW1lKCkgLSB0X2VwKSAvIDYwLCAxKSkKICAgIGcgPSBnb2xkX2V2YWwoZiJlcHtlcH0iKQogICAgc2F2ZShmImVwe2VwfSIsIHsiZXBvY2giOiBlcCwgImdvbGRfbWFjcm8iOiBnfSkKICAgIGlmIHRpbWUudGltZSgpIC0gVDAgPiBidWRnZXRfczoKICAgICAgICBsb2coZXY9ImJ1ZGdldF9zdG9wIiwgZXA9ZXApCiAgICAgICAgYnJlYWsKc2F2ZSgibGFzdCIsIHsiZXBvY2giOiBlcH0pCmxvZyhldj0iZG9uZSIsIGVwb2Noc19ydW49ZXAgKyAxLCBwbGFubmVkPWVwb2NocywgaG91cnM9cm91bmQoKHRpbWUudGltZSgpIC0gVDApIC8gMzYwMCwgMikpCg=="
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
ARMS = {"A": dict(gpu=0, extra=[]), "B": dict(gpu=1, extra=["--T", "12", "--zoom", "0.6"])}
procs = {}
for arm, cfg in ARMS.items():
    out = f"{W}/arm{arm}"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, f"{W}/core/train_ft.py", "--arm", arm, "--core", f"{W}/core", "--parent", parent,
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
