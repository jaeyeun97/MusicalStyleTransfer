mkdir -p fma && cd fma
#curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
#curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
#curl -O https://os.unil.cloud.switch.ch/fma/fma_medium.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_large.zip

echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -
echo "c67b69ea232021025fca9231fc1c7c1a063ab50b  fma_medium.zip"   | sha1sum -c -
echo "497109f4dd721066b5ce5e5f250ec604dc78939e  fma_large.zip"    | sha1sum -c -

unzip fma_metadata.zip
unzip fma_small.zip
unzip fma_medium.zip
unzip fma_large.zip
