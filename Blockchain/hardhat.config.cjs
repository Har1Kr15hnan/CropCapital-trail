require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

// --- THE NUCLEAR FIX ---
// This function ignores quotes, spaces, '0x' prefixes, and garbage.
// It hunts for exactly 64 hex characters (your key) and ignores the rest.
const getCleanKey = () => {
    const raw = process.env.PRIVATE_KEY || "";
    
    // Regex: Find exactly 64 characters of a-f, 0-9
    const match = raw.match(/[a-fA-F0-9]{64}/);
    
    if (match) {
        // Return it with the single necessary 0x prefix
        console.log(`✅ Key found and cleaned. Using: 0x${match[0].substring(0, 4)}...`);
        return "0x" + match[0];
    }
    
    console.error("❌ CRITICAL: No valid 64-char private key found in .env!");
    return "0x0000000000000000000000000000000000000000000000000000000000000000"; // Dummy key to prevent crash
};

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.24",
  networks: {
    amoy: {
      url: "https://rpc-amoy.polygon.technology/",
      accounts: [getCleanKey()],
    },
  },
};