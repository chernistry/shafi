import os
import shutil
import subprocess
from pathlib import Path

# SCT docs found via Qdrant
sct_docs = "44eada8b59109a9786d26b923013ae3674be81e37fc9d5a10c5d2e46f04068f7,fbcf7b546efdab6e21f2023d43ab8edd654c91d75e9e5c7d8bb134a662711699,606129de12e01432648064237207b22367b5fd0d72fd287e5fce6eb394245cfd,1635b7381219c2702f25f11e876b3f6c4e99deef68b3f7d424fa32a81952dd64,2479d3be6e11a85a6337b32e90ffd398d25a1336a12c0ed5f1e915917f29032f,e4741149b9a44c7cd24286313a2c89e92923b8e99786d4b02107f093aa8293c2,3df91d2a0d246b2c70f663cf92e9c0e345ef2485cf7027ccb1c24ab5732158bd,eaa0fe6459acec126535e416342312727d056a9fff5894358e4a47604fd043bd,00427a0402e71479e78abd615764eba7f3874352663aa227caeca830a3cbf6b4,4cccd40fc2bd7aee0287bbff7ac8d922bb4ea4e4646c96af7c16784657d6e884,4e30baac0c5eb8273894d8f7e17d4066356c92eb15eedcc802abf45cf942da43,0a59bb9e8700bdbde4b2cd2ccb3ef6d0e80558a346cf7308f4ff2a0868b1a76c,8f8837a8ccf2a27e9f7264f569a2df49aea92c1d6ecee753c6e307c9c5c37b87,e9a32d22eb9bac59756c1fe25e6d5ffd09d75582efcbe191fe8f078145cc08a5,f4c4d051d514270964adcf58e124569e396602340b7d9172671759de10a95897,65f3f4acfcaca7fc9488e275643135ea7e1d30323e4f4c2cc04d7fc16844e373,df1355930bf24acf13f161965f6f413db2877773a500f4d5a804a1e3f951166e,ea81bf0f8c1d423dd61e6d2901ca7872209185eb8e25e05a672148a9fb3e635e,37f14b6d6d2a99ea50a8cbe6221c0bef5a0b75a65e270e9b1144b6c2acdb2f57,78461539a8b7238c314b7dd56483e1682c5e1ce7f3cf8fc6947e22c0ac5d0d18,09660f78c26cd56c08c7253ed21ba01fb246092f482ccd8acd8e6f9b6fd2d917,ee0796ad2bae4b2eb02e0859f0d655986ffaac811ed763de1ee4fb01d4a66562,3a574fc4f0baa5ec7ff0dcc822d7c55e4f9c493c24addc217c8df3d848c16fa5,e8f40f6102b8f7e2f7e48b43620d9a4b7af36cc231178d42811e404435a4422b,9942e2b7147108ca3342d987ba3ebf09c608a86fb55d05632d56744ed0cda20e,a3dd5a428e8b26001ded74cfa5a8ebf9d3e6fa7c9732bd7cf9995aaba0858f07,e46e9d9e140e3e4dcb8cc2d111edd8d14ba631554d4961790022e13212eeef57,6306079a16b1dec85690f75c715cdbd78b0685a3e19ee30250d481bc32f2e29a,6c3c49e1a35a1e3577a0412f97b6e3d91cae11d24430c7390d7865c992b1a515,e70ae1d37abe6ea09ad1615c2480411ef34b7d0f4b9defc50b63f8815ac79d36,0a18273af2033f502df727480033b00235011b7959142a5bf3deadbca79cfabb,5f90e82bc4d43a8b99c2ee4b227e7acc7410c59dd6fa3b243f7295c56c69aa9d".split(",")

# Bad title docs
bad_title_docs = [
    "96853cbb2873718b7613b42de18dcf417aac607e7fd3e143cab9db7f40622263",
    "31af73c34191903fc3fab3233322ab3e97edb1820a966af313200cdb9bd3b24e",
    "bac066005f21591dbcff19a56cdef279bd4f32482e3758925a38c478e75b81a8",
    "a3ac668ff1a2630ae4826e239487862831558f29e896186631f72ab30a713ac4",
    "0ee112f26f9b33c272a4fad301c52e7309c98236d627ad2451202ca6f5f0cdc7",
    "c850b3f04668abe265441e338a92d3ec1c547948b895cf1c27e4a7f2506bb130",
    "3efee4fe904cb89605ab5baee9f20a97242c4e40e9a496c682448f91c9515497",
    "8309c7e167d19956c92d7f1db1c878e6bbd66afd5eff176e025427e008105867",
    "fb4addc95a4cf4e622a048ba729de434400e523f293197cb979514485c067b1b",
    "f4616ad41ec3ba5b428b3e40c2ede7b5f37f2fe26d30d6fc152db8924bfda10c",
    "e822a7554e725938959c53b53e5e3680c9db959151292423277bd48f6b404cd1",
    "f513711cc3a06447b41a1b798acb1fae36a82ea4d95a0aa3977d4a71ea79d704",
    "d1a939b689b84a9dd683e3dfc93ccb238e310476f9c3249492cf09868852ebd9",
    "77ed6765ad5556f347f1e9fdfb0ce72dba0451713359f73aae3fbc00b35ede32",
    "5c9d86f9661b03024861c781332d81f275b34fc894d2d5bc11f9e0e1cfbb4b93",
    "849bf95cdc57e206fdab9597da97c15e608252902252e4ac6ac3c7cb1db89aa8",
    "324b44750e9690ebf3204312bec975c1b1bca887c0c9d28666722d066e6ae114",
    "fe81efddb24f01f4455fc6b16fc0867615e840b0d36e7abfd1441e5c2eaed92d"
]

all_affected = set(sct_docs) | set(bad_title_docs)
print(f"Total affected docs to re-ingest: {len(all_affected)}")

temp_dir = Path("data/reingest_temp")
temp_dir.mkdir(parents=True, exist_ok=True)

found_count = 0
for doc_id in all_affected:
    # Files are in dataset/dataset_documents/
    src = Path(f"dataset/dataset_documents/{doc_id}.pdf")
    if src.exists():
        shutil.copy(src, temp_dir / src.name)
        found_count += 1
    else:
        # Check in dataset/private/documents too just in case
        src = Path(f"dataset/private/documents/{doc_id}.pdf")
        if src.exists():
            shutil.copy(src, temp_dir / src.name)
            found_count += 1
        else:
            print(f"Warning: Could not find PDF for {doc_id}")

print(f"Copied {found_count} PDFs to {temp_dir}")

# Now run ingestion on the temp dir
# We must set the environment variables from the profile
env_file = "profiles/private_v9_rerank12.env"
print(f"Running ingestion using {env_file}...")

# We use --skip-deletion to keep other docs in Qdrant, but it will overwrite points for these docs
# Wait, pipeline.py --doc-dir will process ALL files in doc-dir.
# QdrantStore.upsert_chunks will overwrite points with same UUID.
# Point UUID is uuid5(NAMESPACE_URL, chunk_id).
# chunk_id is doc_id:page_num:hash.
# If the doc_id is the same, and page_num is same, and content is same, the chunk_id is same.
# So it will overwrite.

cmd = f"export $(grep -v '^#' {env_file} | xargs) && uv run python -m rag_challenge.ingestion.pipeline --doc-dir {temp_dir} --skip-deletion"
subprocess.run(cmd, shell=True, check=True)

print("RE-INGESTION COMPLETE.")
# Clean up
# shutil.rmtree(temp_dir)
