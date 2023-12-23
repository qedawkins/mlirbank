for FILE in $(ls binaries)
do
    echo "Testing binaries/$FILE"
    spirv-val binaries/$FILE
done
