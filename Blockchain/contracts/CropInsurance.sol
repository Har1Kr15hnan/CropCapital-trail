// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract CropInsurance {
    address public owner;
    event ReportRecorded(string reportHash, int256 creditScore, uint256 timestamp);

    constructor() {
        owner = msg.sender;
    }

    function recordCreditScore(string memory _reportHash, int256 _score) public {
        emit ReportRecorded(_reportHash, _score, block.timestamp);
    }
}