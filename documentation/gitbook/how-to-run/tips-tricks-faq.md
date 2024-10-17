---
description: All the little things to save you time on the clusters
---

# Tips, tricks, FAQ

### Deleting `model_output/` (or any big folder) is too long on the cluster

Yes, it takes ages because IO can be so slow, and there are many small files. If you are in a hurry, you can do

```bash
mv model_output/ model_output_old
rm -r model_output_old &
```

The first command rename/move `model_output`, it is instantaneous. You can now re-run something. To delete the renamed folder, run the second command. the `&` at the end makes it execute in the background.
