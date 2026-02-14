const hre = require("hardhat");

async function main() {
  console.log("Deploying CropInsurance contract...");

  // 1. Get the compiled contract
  const CropInsurance = await hre.ethers.getContractFactory("CropInsurance");

  // 2. Deploy it
  const contract = await CropInsurance.deploy();

  // 3. Wait for the transaction to finish
  await contract.waitForDeployment();

  // 4. Print the result
  console.log(`CropInsurance Deployed to: ${contract.target}`);
}

// Error handling
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});